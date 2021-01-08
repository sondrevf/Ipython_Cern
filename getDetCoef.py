import numpy as np

# From DELPHI.py
def detuning_coef_oct_LHC(IOct,teleIndex,gamma):

    current_max=550.

    # reduction factor for the octupole current and the energy
    F=(IOct/current_max)*(7460.52/gamma)
    D=(-IOct/current_max)*(7460.52/gamma)

    # use madx value of O_3 (63100 T/m3) and madx beta functions (checked with S. Fartoukh).
    # We put the same values as in HEADTAIL.

    #ax=9789*F-277203*D        # vertical
    ax=267065*F-7856*D         # horizontal
    axy=-102261*F+93331*D

    # amplification by the teleoptics
    ax*=0.25*(1.0/teleIndex+teleIndex)**2
    axy*=0.5*(1+0.25*(1.0/teleIndex+teleIndex)**2)

    return ax,axy # need to multiply by physical emittance to obtain detunging

if __name__ == '__main__':
    teleIndex = 1.0
    energy = [6.5E3,7.0E3][1]
    gamma = energy / 0.938
    IOct = 1.0
    normEmit = 1.0E-6
    physEmit = normEmit/gamma
    
    ax,axy = detuning_coef_oct_LHC(IOct,teleIndex,gamma)
    print(ax,axy)
    print(ax*physEmit,axy*physEmit)
