API_KEY = "jvOIx9OzxPJzd279mLT54HrKJK3sDEtYhmnCdlwNIXGj2khi3KlL3aDpc4kROHyD"
SECRET_KEY = "WJKs1frrKOyOepzlq06VTcTMRwKaBICcc9aCKJD3L2JMXhhoH3vlpG6Do8ymAgv7"
INTERVAL_1MINUTE = "1m"
INTERVAL_3MINUTE = "3m"
INTERVAL_5MINUTE = "5m"
INTERVAL_15MINUTE = "15m"
INTERVAL_30MINUTE = "30m"
INTERVAL_1HOUR = "1h"
DATABASE_URL_1MINUTE = "sqlite:///database/Crypto_1m.sqlite"
DATABASE_URL_3MINUTE = "sqlite:///database/Crypto_3m.sqlite"
DATABASE_URL_5MINUTE = "sqlite:///database/Crypto_5m.sqlite"
DATABASE_URL_15MINUTE = "sqlite:///database/Crypto_15m.sqlite"
DATABASE_URL_30MINUTE = "sqlite:///database/Crypto_30m.sqlite"
DATABASE_URL_1HOUR = "sqlite:///database/Crypto_1h.sqlite"

time_frame_to_data_base = {
    INTERVAL_1MINUTE: DATABASE_URL_1MINUTE,
    INTERVAL_3MINUTE: DATABASE_URL_3MINUTE,
    INTERVAL_5MINUTE: DATABASE_URL_5MINUTE,
    INTERVAL_15MINUTE: DATABASE_URL_15MINUTE,
    INTERVAL_30MINUTE: DATABASE_URL_30MINUTE,
    INTERVAL_1HOUR: DATABASE_URL_1HOUR,
}

tranding_pairs = ["SOLUSDT", "MATICUSDT", "SANDUSDT", "BTCUSDT", "ETHUSDT"]

pairs = [
    "XRPUSDT",
    "TRXUSDT",
    "WAVESUSDT",
    "ZILUSDT",
    "ONEUSDT",
    "COTIUSDT",
    "SOLUSDT",
    "EGLDUSDT",
    "AVAXUSDT",
    "NEARUSDT",
    "FILUSDT",
    "AXSUSDT",
    "ROSEUSDT",
    "ARUSDT",
    "MBOXUSDT",
    "YGGUSDT",
    "BETAUSDT",
    "PEOPLEUSDT",
    "EOSUSDT",
    "ATOMUSDT",
    "FTMUSDT",
    "DUSKUSDT",
    "IOTXUSDT",
    "OGNUSDT",
    "CHRUSDT",
    "MANAUSDT",
    "XEMUSDT",
    "SKLUSDT",
    "ICPUSDT",
    "FLOWUSDT",
    "WAXPUSDT",
    "FIDAUSDT",
    "ENSUSDT",
    "SPELLUSDT",
    "LTCUSDT",
    "IOTAUSDT",
    "LINKUSDT",
    "XMRUSDT",
    "DASHUSDT",
    "MATICUSDT",
    "ALGOUSDT",
    "ANKRUSDT",
    "COSUSDT",
    "KEYUSDT",
    "XTZUSDT",
    "RENUSDT",
    "RVNUSDT",
    "HBARUSDT",
    "BCHUSDT",
    "COMPUSDT",
    "ZENUSDT",
    "SNXUSDT",
    "SXPUSDT",
    "SRMUSDT",
    "SANDUSDT",
    "SUSHIUSDT",
    "YFIIUSDT",
    "KSMUSDT",
    "DIAUSDT",
    "RUNEUSDT",
    "AAVEUSDT",
    "1INCHUSDT",
    "ALICEUSDT",
    "FARMUSDT",
    "REQUSDT",
    "GALAUSDT",
    "POWRUSDT",
    "OMGUSDT",
    "DOGEUSDT",
    "SCUSDT",
    "XVSUSDT",
    "ASRUSDT",
    "CELOUSDT",
    "RAREUSDT",
    "ADXUSDT",
    "CVXUSDT",
    "WINUSDT",
    "C98USDT",
    "FLUXUSDT",
    "ENJUSDT",
    "FUNUSDT",
    "KP3RUSDT",
    "ALCXUSDT",
    "ETCUSDT",
    "THETAUSDT",
    "CVCUSDT",
    "STXUSDT",
    "CRVUSDT",
    "MDXUSDT",
    "DYDXUSDT",
    "OOKIUSDT",
    "CELRUSDT",
    "RSRUSDT",
    "ATMUSDT",
    "LINAUSDT",
    "POLSUSDT",
    "ATAUSDT",
    "RNDRUSDT",
    "NEOUSDT",
    "ALPHAUSDT",
    "XVGUSDT",
    "KLAYUSDT",
    "DFUSDT",
    "VOXELUSDT",
    "LSKUSDT",
    "KNCUSDT",
    "NMRUSDT",
    "MOVRUSDT",
    "PYRUSDT",
    "ZECUSDT",
    "CAKEUSDT",
    "HIVEUSDT",
    "UNIUSDT",
    "SYSUSDT",
    "BNXUSDT",
    "GLMRUSDT",
    "LOKAUSDT",
    "CTSIUSDT",
    "REEFUSDT",
    "AGLDUSDT",
    "MCUSDT",
    "ICXUSDT",
    "TLMUSDT",
    "MASKUSDT",
    "IMXUSDT",
    "XLMUSDT",
    "BELUSDT",
    "HARDUSDT",
    "NULSUSDT",
    "TOMOUSDT",
    "NKNUSDT",
    "BTSUSDT",
    "LTOUSDT",
    "STORJUSDT",
    "ERNUSDT",
    "XECUSDT",
    "ILVUSDT",
    "JOEUSDT",
    "SUNUSDT",
    "ACHUSDT",
    "TROYUSDT",
    "YFIUSDT",
    "CTKUSDT",
    "BANDUSDT",
    "RLCUSDT",
    "TRUUSDT",
    "MITHUSDT",
    "AIONUSDT",
    "ORNUSDT",
    "WRXUSDT",
    "WANUSDT",
    "CHZUSDT",
    "ARPAUSDT",
    "LRCUSDT",
    "IRISUSDT",
    "UTKUSDT",
    "QTUMUSDT",
    "GTOUSDT",
    "MTLUSDT",
    "KAVAUSDT",
    "DREPUSDT",
    "OCEANUSDT",
    "UMAUSDT",
    "FLMUSDT",
    "UNFIUSDT",
    "BADGERUSDT",
    "PONDUSDT",
    "PERPUSDT",
    "TKOUSDT",
    "GTCUSDT",
    "TVKUSDT",
    "MINAUSDT",
    "RAYUSDT",
    "LAZIOUSDT",
    "AMPUSDT",
    "BICOUSDT",
    "CTXCUSDT",
    "FISUSDT",
    "BTGUSDT",
    "TRIBEUSDT",
    "QIUSDT",
    "PORTOUSDT",
    "DATAUSDT",
    "NBSUSDT",
    "EPSUSDT",
    "TFUELUSDT",
    "BEAMUSDT",
    "REPUSDT",
    "PSGUSDT",
    "WTCUSDT",
    "FORTHUSDT",
    "BONDUSDT",
    "ZRXUSDT",
    "FIROUSDT",
    "SFPUSDT",
    "VTHOUSDT",
    "FIOUSDT",
    "PERLUSDT",
    "WINGUSDT",
    "AKROUSDT",
    "BAKEUSDT",
    "ALPACAUSDT",
    "FORUSDT",
    "IDEXUSDT",
    "PLAUSDT",
    "VITEUSDT",
    "DEGOUSDT",
    "XNOUSDT",
    "STMXUSDT",
    "JUVUSDT",
    "STRAXUSDT",
    "CITYUSDT",
    "JASMYUSDT",
    "DEXEUSDT",
    "OMUSDT",
    "MKRUSDT",
    "FXSUSDT",
    "BTCUSDT",
    "ETHUSDT",
    "ADAUSDT",
]
