from agents.crl import CRLAgent
from agents.dsharsa import DSHARSAAgent
from agents.gcfbc import GCFBCAgent
from agents.gcfql import GCFQLAgent
from agents.gciql import GCIQLAgent
from agents.gcsacbc import GCSACBCAgent
from agents.hgcfbc import HGCFBCAgent
from agents.hiql import HIQLAgent
from agents.ngcsacbc import NGCSACBCAgent
from agents.sharsa import SHARSAAgent
from agents.tmd import TMDAgent
from agents.fqltmd import FQLTMDAgent
from agents.f_tmd import F_TMDAgent
from agents.tmd_new import TMD2Agent

agents = dict(
    crl=CRLAgent,
    dsharsa=DSHARSAAgent,
    gcfbc=GCFBCAgent,
    gcfql=GCFQLAgent,
    gciql=GCIQLAgent,
    gcsacbc=GCSACBCAgent,
    hgcfbc=HGCFBCAgent,
    hiql=HIQLAgent,
    ngcsacbc=NGCSACBCAgent,
    sharsa=SHARSAAgent,
    tmd=TMDAgent,
    fqltmd=FQLTMDAgent,
    f_tmd=F_TMDAgent,
    tmd2=TMD2Agent,
)
