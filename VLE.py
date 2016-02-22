import numpy as np, matplotlib.pyplot as mpl_plt, scipy.optimize as sci_opt, xlrd

# K, Pa
Tc, Pc = 647, 22064000
# 1 atm
P = 101325
mol_water_72 = 0.01622
mol_water_105 = 0.01622
# molar mass
LiCl, H2O = 42.394, 18.01
LiCl_mols = 1
# Pa
P_LiCl = 350


# Input T is in C
def p_vap_water(T):
    A = [-7.585230, 1.839910, -11.781100, 22.670500, -15.939300, 1.755160]
    tau = 1 - (T + 273)/Tc

    if T == 0:
        return 0
    else:
        return Pc * np.exp((A[0]*tau + A[1]*np.power(tau, 1.5) + A[2]*np.power(tau, 3) + A[3]*np.power(tau, 3.5) + A[4]*np.power(tau, 4) + A[5]*np.power(tau, 7.5))/(1-tau))


# Input T is in C
def p_vap_solution(x, T):
    pi = [0.28, 4.3, 0.6, 0.21, 5.1, 0.49, 0.362, -4.75, -0.4, 0.03]
    # convert mass to mole fraction
    w = x*LiCl / (x*LiCl + (1-x)*H2O)
    theta = (T + 273)/Tc

    A = 2 - np.power((1 + np.power((w/pi[0]), pi[1])), pi[2])
    B = np.power((1 + np.power((w/pi[3]), pi[4])), pi[5]) - 1
    f = A + B*theta
    pi25 = 1 - np.power((1 + np.power((w/pi[6]), pi[7])), pi[8]) - pi[9]*np.exp(-np.power((w - 0.1), 2)/0.005)

    return pi25 * f * p_vap_water(T)


def sol_limit(T):

    if -75.5 < T < -68.2:
        A = [-0.005340, 2.015890, -3.114590]
    elif -68.2 < T < -19.9:
        A = [-0.560360, 4.723080, -5.811050]
    elif -19.9 < T < 19.1:
        A = [-0.351220, 2.882480, -2.624330]
    elif 19.1 < T < 93.8:
        A = [-1.312310, 6.177670, -5.034790]
    else:
        A = [-1.356800, 3.448540, 0.0]

    theta = (T + 273) /Tc
    #quadratic formula
    # A[2]x^2 + A[1]x + A[0] = theta
    a, b, c = A[2], A[1], A[0] - theta
    w = (-b + np.sqrt(np.power(b, 2) - 4*a*c)) / (2*a)
    limit = w*(1/LiCl) / (w*(1/LiCl) + (1-w)*(1/H2O))

    return limit


def convert_F_to_C(T):
    return (T - 32) * 5/9


def convert_RH_to_Pa(RH, T):
    return RH * p_vap_water(T)


# Input T is in F
def plot_VLE(Tin, Tout, Treg, RHin, RHout):
    Tin, Tout, Treg = convert_F_to_C(Tin), convert_F_to_C(Tout), convert_F_to_C(Treg),

    #linspace(0,1,100) generates 100 numbers between 0 and 1
    x_range_Tin = np.linspace(0.00001, sol_limit(Tin), 100)
    y_range_Tin = p_vap_solution(x_range_Tin, Tin)
    mpl_plt.plot(x_range_Tin, y_range_Tin, color = 'b', label = "Indoor")
    mpl_plt.plot(x_range_Tin[99], y_range_Tin[99], color = 'b', marker = 'o')

    # Creates straight line between two points
    x_range_RHin = (0, sol_limit(Tin))
    y_range_RHin = (convert_RH_to_Pa(RHin, Tin), convert_RH_to_Pa(RHin, Tin))
    mpl_plt.plot(x_range_RHin, y_range_RHin, linestyle = '--', label = "Indoor R.H")

    x_range_Treg = np.linspace(0.00001, sol_limit(Treg), 100)
    y_range_Treg = p_vap_solution(x_range_Treg, Treg)
    mpl_plt.plot(x_range_Treg, y_range_Treg, color = 'g', label = "Regenerator")
    mpl_plt.plot(x_range_Treg[99], y_range_Treg[99], color = 'g', marker = 'o')

    x_range_RHout = (0, sol_limit(Treg))
    y_range_RHout = (convert_RH_to_Pa(RHout, Tout), convert_RH_to_Pa(RHout, Tout))
    mpl_plt.plot(x_range_RHout, y_range_RHout, linestyle = '--', label = "Outdoor R.H")

    mpl_plt.legend()
    mpl_plt.xlabel("LiCl mole fraction")
    mpl_plt.ylabel("Water Vapor Pressure (Pa)")
    mpl_plt.show()

    return


# T1 = operating, T2 = indoor/outdoor (only different for regen/outdoor)
def get_mole_fraction(T1, T2, RH, guess):
    T1, T2 = convert_F_to_C(T1), convert_F_to_C(T2)
    f = lambda x: p_vap_solution(x, T1) - convert_RH_to_Pa(RH, T2)

    return sci_opt.fsolve(f, guess)


def find_operating_range(Tin, Tout, Treg, RHin, RHout, guess):
    x_low = get_mole_fraction(Tin, Tin, RHin, guess)
    x_hi = get_mole_fraction(Treg, Tout, RHout, guess)

    return [x_low, x_hi]


def get_k(x, T):
    P_sat = p_vap_solution(x, T)
    return P_sat / P


def find_moles_of_water(RH, T, guess):
    f = lambda SH: (SH * P) / (((0.622 + 0.378 * SH) * p_vap_water(T)) * RH) - 1

    return sci_opt.fsolve(f, guess) * 29 / 18


def find_water_absorbed(x_in, RH, T):
    T = convert_F_to_C(T)
    K = get_k(x_in, T)
    # 50 mol/s basis for air
    V_in = 50
    # 1 mol/s basis for LiCl -> L_in * x_in = 1
    L_in = LiCl_mols / x_in
    A = L_in/(K*V_in)
    percent_abs = (np.power(A, 2) - A) / (np.power(A, 2) - 1)
    return V_in * find_moles_of_water(RH, T, .01) * percent_abs


def find_water_absorbed_alt(x_in, x_limit):
    return np.abs((LiCl_mols / x_in) - (LiCl_mols / x_limit))


def plot_water_absorbed():
    range = find_operating_range(72, 85, 105, .6, .8, .1)
    x_range = np.linspace(range[0], range[1], 100)
    y_range = find_water_absorbed(x_range, .6, 72)

    mpl_plt.plot(x_range, y_range)
    mpl_plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    mpl_plt.xlabel("LiCl mole fraction")
    mpl_plt.ylabel("Water Absorbed (mol/s per mol/s of LiCl)")
    mpl_plt.savefig("Water_Absorbed")
    mpl_plt.show()

    return


def plot_water_absorbed_alt():
    range = find_operating_range(72, 85, 105, .6, .8, .1)
    x_range = np.linspace(range[0], range[1], 100)
    y_range = find_water_absorbed_alt(x_range, range[0])

    mpl_plt.plot(x_range, y_range)
    mpl_plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    mpl_plt.xlabel("LiCl mole fraction")
    mpl_plt.ylabel("Water Removed (mol/s per mol/s of LiCl)")
    mpl_plt.savefig("Water_Absorbed_Alt")
    mpl_plt.show()

    return


def find_power_req(x):

    #Initialize constants
    Mw_L = 42.39
    Mw_H = 18.02
    T_F = 313.71
    T_I = 295.37
    L = LiCl_mols / x

    #Average molecular weight
    Mw_avg = x * Mw_L + (1 - x) * Mw_H

    #Converted value of heat capacity to mole basis (KJ/(mol*K)
    Cp = 3.07 / 1000 * Mw_avg

    #Power in KW
    power = L * Cp * (T_F - T_I)

    return power


def plot_power():
    range = find_operating_range(72, 85, 105, .6, .8, .1)
    x = np.linspace(range[0], range[1], 100)
    y_range = find_power_req(x)

    mpl_plt.plot(x, y_range)
    mpl_plt.xlabel("LiCl mole fraction")
    mpl_plt.ylabel("Power Required (kW)")
    mpl_plt.savefig("Power_Required.png")
    mpl_plt.show()

    return


def find_maximum_water_removed():
    range = find_operating_range(72, 85, 105, .6, .8, .1)
    x_range = np.linspace(range[0], range[1], 100)
    y_range = find_water_absorbed(x_range, .6, 72)

    return np.amax(y_range)


def find_maximum_water_removed_alt():
    range = find_operating_range(72, 85, 105, .6, .8, .1)
    x_range = np.linspace(range[0], range[1], 100)
    y_range = find_water_absorbed_alt(x_range, range[0])

    return np.amax(y_range)


def graph_water_absorbed_excel():
    data = np.loadtxt(open("data1.csv","rb"),delimiter=",")
    x_range = data[0]
    y_range = data[1]
    mpl_plt.plot(x_range, y_range)
    mpl_plt.xlabel("LiCl mole fraction")
    mpl_plt.ylabel("Water Removed (mol/s per mol/s of LiCl)")
    mpl_plt.savefig("Water_Absorbed_Exc")
    mpl_plt.show()


def graph_power_excel():
    data = np.loadtxt(open("data2.csv","rb"),delimiter=",")
    x_range = data[0]
    y_range = data[1]
    mpl_plt.plot(x_range, y_range)
    mpl_plt.xlabel("LiCl mole fraction")
    mpl_plt.ylabel("Power Required (kW)")
    mpl_plt.savefig("Power_Required_Exc.png")
    mpl_plt.show()


def plot_time_alt():
    range = find_operating_range(72, 85, 105, .6, .8, .1)
    x_range = np.linspace(range[0], range[1], 100)
    y_range = (170 / find_water_absorbed_alt(x_range, range[0])) / 60

    # Creates straight line between two points
    x_range_time = (range[0], range[1])
    y_range_time = (10, 10)
    mpl_plt.plot(x_range_time, y_range_time, linestyle = '--')

    mpl_plt.plot(x_range, y_range)
    mpl_plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    mpl_plt.xlabel("LiCl mole fraction")
    mpl_plt.ylabel("Time Required")
    mpl_plt.savefig("Time_Alt")
    mpl_plt.show()

    return


def convert_x_to_X(x):
    return x / (1 - x)


def plot_VLE_XY(Tin, Tout, Treg, RHin, RHout):
    range = find_operating_range(Tin, Tout, Treg, RHin, RHout, .1)

    Tin, Tout, Treg = convert_F_to_C(Tin), convert_F_to_C(Tout), convert_F_to_C(Treg)
    x_range = np.linspace(range[0], range[1], 100)
    y_range = p_vap_solution(x_range, Tin) / P

    X = (1 - convert_x_to_X(x_range))
    Y = convert_x_to_X(y_range)

    # Creates straight line between two points
    X_RH = (X[0], X[99])
    y_RH = convert_RH_to_Pa(RHin, Tin) / P
    Y_RH = (convert_x_to_X(y_RH), convert_x_to_X(y_RH))

    mpl_plt.plot(X, Y, label = "Equilibrium Line")
    mpl_plt.plot(X_RH, Y_RH, linestyle = '--', label = "Indoor R.H")


    mpl_plt.xlabel("H2O liquid mole ratio")
    mpl_plt.ylabel("H2O vapor mole Ratio")
    mpl_plt.savefig("VLE_XY")
    mpl_plt.show()


def solve_abs(Y_1, N):
    range = find_operating_range(72, 85, 105, .6, .8, .1)

    X = [(1 - convert_x_to_X(range[0]))[0], (1 - convert_x_to_X(range[1]))[0]]
    Y = [convert_x_to_X(p_vap_solution(range[0], convert_F_to_C(72)) / P)[0], convert_x_to_X(p_vap_solution(range[1], convert_F_to_C(72))/ P)[0]]

    eq_coeff = np.polyfit(X, Y, 1)
    eq_line = np.poly1d(eq_coeff)

    y_RH = convert_RH_to_Pa(0.6, convert_F_to_C(72)) / P
    Y_RH = convert_x_to_X(y_RH)

    X_N = X[1]
    Y_N_1 = Y_RH
    X_0 = X[0]

    op_coeff = np.polyfit([X_0, X_N], [Y_1, Y_N_1], 1)
    op_line = np.poly1d(op_coeff)

    counter = 0




# #1.1
# plot_VLE(72, 85, 105, .6, .8)
# plot_water_absorbed()
# plot_water_absorbed_alt()
# print(find_operating_range(72, 85, 105, .6, .8, .1))
#
# #1.2
# plot_power()
#
# #1.3
# print(find_maximum_water_removed())
# print(find_maximum_water_removed_alt())
#
# graph_water_absorbed_excel()
# graph_power_excel()

# print(find_moles_of_water(0.6, convert_F_to_C(72), .1))
# plot_VLE_XY(72, 85, 105, .6, .8)
solve_abs(0, 0)