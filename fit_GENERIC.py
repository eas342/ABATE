from abate import simple_pymc_fits
import urllib
import os
import shutil
from tshirt.pipeline import phot_pipeline
homedir = os.getcwd()
direct = 0

def do_fitting():
    while True:
        try:
            paramPath = 'parameters/phot_params/jwst_test_data/sim_mirage_038_hatp14_no_backg_source/phot_param_mirage_038_no_backg_source.yaml'
            mod = simple_pymc_fits.exo_model(paramPath=paramPath,
                                             descrip='mirage_038_hatp14_p001_A3_phot_no_backgsrc_starry_003_fix_LD',t0_lit=(2459651.04,0.03),
                                             period_lit=(4.627664, 2e-6),
                                             inc_lit=(83.5, 1.0),
                                             a_lit=(8.87,0.1),
                                             u_lit=[0.05678697,  0.63804229, -2.7843976 ,  6.12289066, -6.14746493, 2.37283885],
                                             starry_ld_degree=6,
                                             ecc=(0.1074,0.01),
                                             omega=(94.4,0.5),
                                             recalculateTshirt=True,
                                             pipeType='phot',ld_law='multiterm')
        except FileNotFoundError as error:
            print(error)
            print('Retrieving File...')
            split1 = str(error).split('/')
            filename_ext = split1[-1].strip("'")
            split2 = filename_ext.split('.')
            filename = split2[0]
            #print(filename)

            if (str(split2[1]) == 'yaml'):
                direct = homedir
                for i in range((len(split1) - 2)):
                    i += 1
                    direct = str(direct + '/' + split1[i])
                #print(direct)
            elif (str(split2[1]) != 'yaml'):
                i = 1
                direct = ''
                while i < int(len(split1) - 1):
                    #print(i, split1[i])
                    direct = direct + '/' + split1[i]
                    i += 1
                #print(direct)

            try:
                os.makedirs((direct + '/'))
            except OSError as error:
                pass

            f = urllib.request.urlopen('https://zenodo.org/record/6629582/files/{}'.format(filename_ext))
            with open('{}'.format(filename_ext),'wb') as f_out:
                f_out.write(f.read())
            shutil.move(str(filename_ext), str(direct + '/' + filename_ext))    

        else:
            mod.run_all_broadband()
            break

