#!/usr/bin/env python3
"""Ticker -> sector/category classification for the joint trading platform.

Two different problems, handled differently:

1. Individual equities (alpaca_trader, aggressive_challenger_trader): real GICS
   sector, from known company identity. High confidence, ~19 names.

2. fintech_trader's ~497 holdings are almost entirely ETFs, not single-name
   equities. GICS sector doesn't apply to a fund. Instead this uses a broad
   asset-class/style taxonomy (US equity - broad, US equity - sector, factor,
   international developed/emerging, fixed income by type, commodities, real
   estate, thematic, crypto-linked, cash-equivalent).

Anything not confidently identified is left "Unclassified" rather than guessed.
That is a deliberate, visible gap in the dashboard, not a bug: fabricating a
sector for a name we don't actually recognize would misstate concentration risk.
"""

# ---------------------------------------------------------------------------
# 1. Individual equities -> real GICS sector
EQUITY_SECTOR = {
    "AAPL": "Information Technology",
    "ABBV": "Health Care",
    "ALAB": "Information Technology",   # Astera Labs, semiconductors
    "AMD": "Information Technology",
    "ARM": "Information Technology",
    "ASTS": "Communication Services",   # AST SpaceMobile, satellite comms
    "AVGO": "Information Technology",
    "BMNR": "Financials",               # Bitmine Immersion, digital-asset treasury/mining
    "CAT": "Industrials",
    "CIEN": "Information Technology",
    "CIFR": "Financials",               # Cipher Mining, bitcoin mining
    "COST": "Consumer Staples",
    "FN": "Information Technology",     # Fabrinet
    "GOOGL": "Communication Services",
    "HL": "Materials",                  # Hecla Mining
    "IBM": "Information Technology",
    "INSM": "Health Care",              # Insmed
    "IONQ": "Information Technology",   # quantum computing
    "IREN": "Information Technology",   # Iris Energy, data centers/bitcoin mining
    "JPM": "Financials",
    "META": "Communication Services",
    "MRVL": "Information Technology",
    "MSFT": "Information Technology",
    "NFLX": "Communication Services",
    "NOK": "Information Technology",
    "NVDA": "Information Technology",
    "OPEN": "Real Estate",              # Opendoor
    "RTX": "Industrials",
    "SNDK": "Information Technology",   # SanDisk
    "SNOW": "Information Technology",
    "SONY": "Consumer Discretionary",
    "TER": "Information Technology",    # Teradyne
    "TSLA": "Consumer Discretionary",
    "UNH": "Health Care",
    "V": "Financials",
    "XOM": "Energy",
    "AAOI": "Information Technology",   # Applied Optoelectronics
    "CELH": "Consumer Staples",         # Celsius Holdings
}
# SGOV is an ETF (0-3mo T-bill) even though alpaca_trader holds it as a cash
# parking spot; classify it with the fund taxonomy below, not as an equity.

# ---------------------------------------------------------------------------
# 2. Fund taxonomy for fintech_trader's ETF book
CATEGORY = {}

def tag(cat, *tickers):
    for t in tickers:
        CATEGORY[t] = cat

# --- US equity, broad market ---
tag("US Equity - Broad Market",
    "ACWV","AOR","AVLV","AVUS","BBCA","BBUS","BINC","BKLC","CGCP","CGGO","CGGR",
    "CGMS","CGMU","CGUS","CGXU","COWG","DFAC","DFAT","DFAU","DFUS","DIA","DIHP",
    "DUHP","DYNF","EQWL","ESGU","ESGV","FTCS","FWD","GARP","GSLC","IOO","ITOT",
    "IUSB","IVV","IWB","IWV","JIRE","JIVE","JQUA","LRGF","MAGS","MGC","MOAT",
    "MTUM","OEF","ONEQ","QQQE","QQQM","QUAL","RSP","SCHB","SCHK","SCHX","SPLV",
    "SPTM","SPUS","SPY","SPYM","TCAF","URTH","VFLO","VONE","VOO","VT","VTI",
    "VTWO","VV","XMHQ","XOVR","FBCG","OMFL","XLG")

# --- US equity, factor / style (value, growth, dividend, quality, small/mid) ---
tag("US Equity - Style/Factor",
    "AVUV","AVDE","AVDV","AVEM","CALF","CGDV","DFAI","DFAR","DFAS","DFAV" ,
    "DFCF","DFEM","DFGR","DFIC","DFIS","DFIV","DFLV","DFSD","DFSV","DFUV",
    "DGRO","DISV","DIVO","DSI","DVY","FALN","FBCG","FDVV","FNDA","FNDF","FNDX",
    "FTHI","FRDM","HDV","IJH","IJJ","IJK","IJR","IJS","IJT","IUSG","IUSV",
    "IVE","IVLU","IVW","IWC","IWD","IWF","IWM","IWN","IWO","IWP","IWR","IWS",
    "IWX","IWY","JAVA","JGRO","JHMM","JPLD","JQUA","KNG","LVHI","MDY","MGK",
    "MGV","NOBL","PPLT","PRF","PVAL","RDVI","RDVY","REM","RPG","RPV","SCHA",
    "SCHD","SCHG","SCHM","SCHV","SCYB","SDVY","SDY","SLYG","SLYV","SMLF",
    "SMMD","SPHB","SPHD","SPHQ","SPLB","SPMD","SPMO","SPSM","SPYD","SPYG",
    "SPYV","VB","VBK","VBR","VIG","VIGI","VLUE","VO","VOE","VOT","VOOG","VOOV",
    "VONG","VONV","VTV","VUG","VYM","VYMI","XMMO","XSMO")

# --- US equity, sector (with the actual GICS sector as the label) ---
tag("US Equity - Sector: Technology",
    "AIQ","BOTZ","BUG","CIBR","DTCR","FDN","FENY","FTEC","FTXL","IAI","IGM",
    "IGV","IVES","IYW","QTUM","SKYY","SMH","SOXQ","SOXX","VGT","XLK","XNTK",
    "XSD","IHI","ARKQ","ARKW")
tag("US Equity - Sector: Health Care",
    "FHLC","IBB","IHI","IYH","PPH","PSI","VHT","XBI","XLV","ARKG")
tag("US Equity - Sector: Financials",
    "IAT","IYF","JEPI","JEPQ","KBE","KBWB","KIE","KRE","VFH","XLF")
tag("US Equity - Sector: Energy",
    "AMLP","AMU","COPX","FENY","IXC","IYE","MLPA","MLPX","PICK","SIL","SILJ",
    "URA","VDE","XLE","XME","PDBC")
tag("US Equity - Sector: Industrials",
    "IYT","ITA","PAVE","PPA","VIS","XAR","XLI")
tag("US Equity - Sector: Materials",
    "IYM","VAW","XLB")
tag("US Equity - Sector: Consumer Discretionary",
    "FDN","IYC","JETS","PEJ","RTH","VCR","XLY","XRT")
tag("US Equity - Sector: Consumer Staples",
    "IYK","VDC","XLP")
tag("US Equity - Sector: Utilities",
    "IDU","UTES","XLU")
tag("US Equity - Sector: Real Estate",
    "ICF","IYR","REET","REM","RWR","SCHH","USRT","VNQ","VNQI","XLRE")
tag("US Equity - Sector: Communication Services",
    "FCOM","IXP","IYZ","VOX","XLC")
tag("US Equity - Sector: Aerospace/Defense",
    "ITA","PPA","XAR")

# --- Thematic / innovation ---
tag("Thematic/Innovation",
    "ARKK","ARKX","AIRR","BLOK","IRBO","LIT","NLR","PICK","QQQJ","ROBO",
    "SNPE","THRO","XOVR","BUG","BOTZ","SKYY")

# --- International developed ---
tag("International Equity - Developed",
    "ACWX","BBAX","BBEU","BBJP","DBEF","ECH","EFA","EFAV","EFG","EFV","EPP",
    "ESGD","EWA","EWC","EWG","EWH","EWI","EWJ","EWL","EWP","EWQ","EWS","EWT",
    "EWU","EZU","FLIN","FLJP","FLKR","GSIE","HEFA","IDEV","IDMO","IEFA","IEUR",
    "IEV","IFRA","IGF","IMTM","IQLT","SCHF","SCZ","SPDW","VEA","VEU","VGK",
    "VPL","VSGX","VXUS")
tag("International Equity - Emerging",
    "AAXJ","ARGT","AVEM","BBCA","BBJP","DFAE","EEM","EEMV","EIDO","EMXC",
    "EPI","EPOL","EWZ","EZA","FLKR","FLMI","GSIE","IEMG","ILF","INDA","MCHI",
    "SCHE","SPEM","VWO","FXI")

# --- Fixed income ---
tag("Fixed Income - Broad/Aggregate",
    "AGG","BND","BNDX","EAGG","GVI","IUSB","JMST","SCHZ","SPAB","TOTL","USIG",
    "VUSB","DBMF")
tag("Fixed Income - Treasury/Government",
    "BIL","BILS","BOXX","EDV","GBIL","GOVT","IEF","IEI","JAAA","SCHO","SCHR",
    "SHV","SHY","SPTI","STIP","SUB","TFLO","TIP","VGIT","VGSH","VTES","VTIP")
tag("Fixed Income - Corporate/High Yield",
    "ANGL","CLOA","FALN","FLOT","FLRN","FLTR","HYD","HYG","HYLB","HYMB","IGIB",
    "IGLB","IGSB","JBBB","JMBS","JMUB","JNK","JPIE","JPST","LMBS","LQD","MBB",
    "MUB","PFF","PFFA","PZA","SCHI","SCYB","SHYG","SPBO","SPIB","SPLB","SPMB",
    "SRLN","USHY","VCIT","VCLT","VCSH","VMBS","VRP","IBDR","IBDS","IBDT","IBDU",
    "IBDV","BSCQ","BSCR","BSCS","BSV","ICVT","PYLD","CWB","CLIP","FIXD")
tag("Fixed Income - Municipal",
    "CMF","HYD","HYMB","MUB","PZA","SUB","VTES")
tag("Fixed Income - Emerging Market Debt",
    "EMB","EMLC")

# --- Commodities ---
tag("Commodities - Precious Metals",
    "AAAU","GDX","GDXJ","GLTR","GNR","HGER","OUNZ","PAAA","PPLT","SGOL","SIL",
    "SILJ","SIVR","SLVP","SLVR")
tag("Commodities - Broad/Diversified",
    "BCI","GNR","HGER","PDBC")

# --- Crypto-linked ---
tag("Crypto-Linked", "BITO", "BTC")

# --- Multi-asset / alternative ---
tag("Multi-Asset/Alternative", "CTA", "DBMF", "BIZD", "USHY", "SHLD")

# --- Cash-equivalent / ultra-short ---
tag("Cash-Equivalent / Ultra-Short",
    "BIL","BILS","BOXX","GBIL","JPST","SGOV","SHV","TFLO","SGOV")

# --- Global (developed + emerging blended, not US-only) ---
tag("Global Equity - All Country", "ACWI", "VXF")

# --- second pass: names missed by the first taxonomy sweep, checked individually ---
tag("International Equity - Emerging", "AIA", "EWW", "EWY", "ESGE", "FNDE")
tag("International Equity - Developed", "DFAX", "EUFN", "FEZ", "IDV", "IXUS", "SCHC", "SCHY")
tag("Thematic/Innovation", "ARTY", "UFO")
tag("Fixed Income - Broad/Aggregate", "BIV", "FBND", "ISTB", "JCPB")
tag("Fixed Income - Corporate/High Yield", "BKLN", "FPE", "FTSL", "SPHY")
tag("Fixed Income - Treasury/Government", "SCHP")
tag("Fixed Income - Emerging Market Debt", "VWOB")
tag("US Equity - Sector: Utilities", "FUTY", "VPU")
tag("US Equity - Sector: Technology", "IXN", "RSPT")
tag("US Equity - Sector: Health Care", "IXJ")
tag("US Equity - Style/Factor", "COWZ", "RWL", "USMV")


def classify(symbol: str) -> tuple[str, bool]:
    """Return (label, is_confident)."""
    if symbol in EQUITY_SECTOR:
        return EQUITY_SECTOR[symbol], True
    if symbol in CATEGORY:
        return CATEGORY[symbol], True
    return "Unclassified", False


if __name__ == "__main__":
    import json, sys
    all_syms = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else []
    out = {}
    unclassified = []
    for s in all_syms:
        label, ok = classify(s)
        out[s] = label
        if not ok:
            unclassified.append(s)
    print(f"classified {len(all_syms) - len(unclassified)}/{len(all_syms)}")
    print("unclassified:", unclassified)
    json.dump(out, open("/tmp/joint/sector_classification.json", "w"), indent=1)
