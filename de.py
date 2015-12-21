import instances.ms
from problem import *
msp = WikiProblem(instances.ms.inst)
msp.cluster(['microsoft'],['engines/links','prop-google'], 2)
msp.discovery()
