import PyTDC as tdc
from tdc.single_pred import ADME
data = ADME(name = 'Caco2_Wang')
df = data.get_data()