"""Stage 0 - read TESS light-curve FITS files (SPOC *_lc.fits, QLP *_llc.fits).

Everything later in the pipeline needs, per cadence:
    time, raw flux (SAP), NASA-processed flux (PDCSAP/KSPSAP, only used for
    the ablation baseline), background, centroid x/y and pointing x/y.
SPOC and QLP name these columns differently; this module hides that.
"""
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits

# column names: canonical -> candidates, first one found wins
_COLS = {
    "time": ["TIME"],
    "sap": ["SAP_FLUX"],
    "pdc": ["PDCSAP_FLUX", "KSPSAP_FLUX", "DET_FLUX"],
    "bkg": ["SAP_BKG"],
    "cx": ["MOM_CENTR1", "SAP_X"],
    "cy": ["MOM_CENTR2", "SAP_Y"],
    "px": ["POS_CORR1"],
    "py": ["POS_CORR2"],
    "quality": ["QUALITY"],
}
AUX = ("bkg", "cx", "cy", "px", "py")


@dataclass
class LightCurve:
    tic: int
    sector: np.ndarray            # per-cadence sector number (for stitched curves)
    time: np.ndarray
    sap: np.ndarray               # crowding-corrected raw flux, normalised per sector
    pdc: np.ndarray               # NASA-processed flux, normalised per sector
    aux: dict = field(default_factory=dict)   # bkg, cx, cy, px, py (NaN-filled if missing)
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return self.time.size

    def select(self, m):
        return LightCurve(self.tic, self.sector[m], self.time[m], self.sap[m], self.pdc[m],
                          {k: v[m] for k, v in self.aux.items()}, dict(self.meta))


def _get(data, key):
    for c in _COLS[key]:
        if c in data.columns.names:
            return np.asarray(data[c], float)
    return None


def read_lc(path):
    """Read one FITS light curve -> LightCurve (bad cadences removed)."""
    with fits.open(path, memmap=False) as h:
        d = h[1].data
        hdr0, hdr1 = h[0].header, h[1].header
        t = _get(d, "time")
        sap = _get(d, "sap")
        pdc = _get(d, "pdc")
        q = _get(d, "quality")
        aux = {k: _get(d, k) for k in AUX}
    n = t.size
    pdc = pdc if pdc is not None else np.full(n, np.nan)
    aux = {k: (v if v is not None else np.full(n, np.nan)) for k, v in aux.items()}
    good = np.isfinite(t) & np.isfinite(sap) & (sap > 0)
    if q is not None:
        good &= q == 0
    # telemetry must be finite where it exists at all
    for k, v in aux.items():
        if np.isfinite(v).any():
            good &= np.isfinite(v)
    t, sap, pdc = t[good], sap[good], pdc[good]
    aux = {k: v[good] for k, v in aux.items()}
    o = np.argsort(t, kind="stable")          # a few files have unsorted time stamps
    t, sap, pdc = t[o], sap[o], pdc[o]
    aux = {k: v[o] for k, v in aux.items()}

    # Undo aperture contamination exactly like SPOC PDC does. Without this every
    # depth measured on SAP is diluted by (1 - CROWDSAP)  (37% on TOI-907!)
    crowd = float(hdr1.get("CROWDSAP", 1.0) or 1.0)
    flfrc = float(hdr1.get("FLFRCSAP", 1.0) or 1.0)
    if sap.size:
        med = np.median(sap)
        sap = (sap - (1 - crowd) * med) / flfrc
        sap = sap / np.median(sap)
        if np.isfinite(pdc).any():
            pdc = pdc / np.nanmedian(pdc)
    sector = int(hdr0.get("SECTOR", -1))
    tic = int(hdr0.get("TICID", hdr0.get("OBJECT", "0").split()[-1]))
    meta = dict(crowdsap=crowd, flfrcsap=flfrc,
                teff=hdr0.get("TEFF"), radius=hdr0.get("RADIUS"),
                logg=hdr0.get("LOGG"), tmag=hdr0.get("TESSMAG"),
                cadence_days=float(np.median(np.diff(t))) if t.size > 1 else np.nan)
    return LightCurve(tic, np.full(t.size, sector), t, sap, pdc, aux, meta)


def stitch(lcs):
    """Concatenate several sectors of one star (each already normalised)."""
    lcs = [l for l in lcs if len(l) > 100]
    if not lcs:
        return None
    o = np.argsort([l.time[0] for l in lcs])
    lcs = [lcs[i] for i in o]
    cat = lambda f: np.concatenate([f(l) for l in lcs])
    meta = dict(lcs[0].meta)
    meta["sectors"] = sorted({int(l.sector[0]) for l in lcs})
    return LightCurve(lcs[0].tic, cat(lambda l: l.sector), cat(lambda l: l.time),
                      cat(lambda l: l.sap), cat(lambda l: l.pdc),
                      {k: cat(lambda l, k=k: l.aux[k]) for k in AUX}, meta)


def bin_lc(lc, minutes=2.0):
    """Bin 20-s fast cadence to 2 min so every file has the same cadence."""
    if lc.meta.get("cadence_days", 0) * 1440 >= minutes * 0.9:
        return lc
    w = minutes / 1440
    k = np.floor((lc.time - lc.time[0]) / w).astype(int)
    _, idx, cnt = np.unique(k, return_index=True, return_counts=True)
    red = lambda y: np.add.reduceat(y, idx) / cnt
    out = LightCurve(lc.tic, lc.sector[idx], red(lc.time), red(lc.sap), red(lc.pdc),
                     {a: red(v) for a, v in lc.aux.items()}, dict(lc.meta))
    out.meta["cadence_days"] = w
    return out
