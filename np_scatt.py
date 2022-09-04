import numpy as np
import gvar as gv

def read_np_data():
    # files to import
    files = [
        "np_data/np_daub_2013.dat",
        "np_data/np_koester_1990.dat",
        "np_data/np_houk_1971.dat",
        "np_data/np_kirilyuk_1987.dat",
        "np_data/np_larson_1980.dat"]

    # load data
    Ene_data  = []  # neutron lab frame kinetic energy (keV)
    Sig_data  = []  # total cross section (b)
    dSig_data = []  # error in total cross section (b)

    for file in files:
        print('reading: ', file)
        data = np.loadtxt(open(file,"r"))
        Ene_data.append(data[:,0])
        Sig_data.append(data[:,1])
        dSig_data.append(data[:,2])

    Ene_data = np.concatenate( Ene_data, axis=0 )
    Sig_data = np.concatenate( Sig_data, axis=0 )
    dSig_data = np.concatenate( dSig_data, axis=0 )

    Sig_gv = gv.gvar(Sig_data, dSig_data)

    print("\nn_data = ", Ene_data.size)
    return Ene_data, Sig_gv
