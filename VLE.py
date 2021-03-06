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
    mpl_plt.savefig("Images/Water_Absorbed")
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
    mpl_plt.savefig("Images/Water_Absorbed_Alt")
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
    mpl_plt.savefig("Images/Power_Required.png")
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
    mpl_plt.savefig("Images/Water_Absorbed_Exc")
    mpl_plt.show()


def graph_power_excel():
    data = np.loadtxt(open("data2.csv","rb"),delimiter=",")
    x_range = data[0]
    y_range = data[1]
    mpl_plt.plot(x_range, y_range)
    mpl_plt.xlabel("LiCl mole fraction")
    mpl_plt.ylabel("Power Required (kW)")
    mpl_plt.savefig("Images/Power_Required_Exc.png")
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
    mpl_plt.savefig("Images/Time_Alt")
    mpl_plt.show()

    return


def convert_x_to_X(x):
    return x / (1 - x)


def plot_VLE_XY_abs(Tin, Tout, Treg, RHin, RHout, Y_1, N):
    range = find_operating_range(Tin, Tout, Treg, RHin, RHout, .1)

    Tin, Tout, Treg = convert_F_to_C(Tin), convert_F_to_C(Tout), convert_F_to_C(Treg)
    x_range = np.linspace(range[0], range[1], 100)
    y_range = p_vap_solution(x_range, Tin) / P

    X_eq = (1 - convert_x_to_X(x_range))
    Y_eq = convert_x_to_X(y_range)

    # Creates straight line between two points
    X_RH = (X_eq[0], X_eq[99])
    y_RH = convert_RH_to_Pa(RHin, Tin) / P
    Y_RH = (convert_x_to_X(y_RH), convert_x_to_X(y_RH))

    X_N = X_eq[0]
    Y_N_1 = Y_RH[0]
    X_0 = X_eq[99]

    X_op = (X_0, X_N)
    Y_op = (Y_1, Y_N_1)

    stages = solve_abs(Y_1, N)
    X_stage = stages[0]
    Y_stage = stages[1]

    mpl_plt.plot(X_eq, Y_eq, label = "Equilibrium Line")
    mpl_plt.plot(X_RH, Y_RH, linestyle = '--', label = "Indoor R.H")
    mpl_plt.plot(X_op, Y_op, label = "Operating Line")
    mpl_plt.plot(X_stage, Y_stage, '-o')

    mpl_plt.legend(loc='lower right')
    mpl_plt.xlabel("H2O liquid mole ratio")
    mpl_plt.ylabel("H2O vapor mole ratio")
    mpl_plt.savefig("Images/VLE_XY_abs_" + str(Y_1) + "_" + str(N) + ".png")
    mpl_plt.show()


def solve_abs(Y_1, N):
    range = find_operating_range(72, 85, 105, .6, .8, .1)

    X_eq = [(1 - convert_x_to_X(range[0]))[0], (1 - convert_x_to_X(range[1]))[0]]
    Y_eq = [convert_x_to_X(p_vap_solution(range[0], convert_F_to_C(72)) / P)[0], convert_x_to_X(p_vap_solution(range[1], convert_F_to_C(72))/ P)[0]]

    # function of (Y) -> x = my + b
    eq_coeff = np.polyfit(Y_eq, X_eq, 1)
    eq_line = np.poly1d(eq_coeff)

    y_RH = convert_RH_to_Pa(0.6, convert_F_to_C(72)) / P
    Y_RH = convert_x_to_X(y_RH)

    X_N = X_eq[1]
    Y_N_1 = Y_RH
    X_0 = X_eq[0]

    op_coeff = np.polyfit([X_N, X_0], [Y_1, Y_N_1], 1)
    op_line = np.poly1d(op_coeff)

    counter = 0
    i = 0
    X = []
    Y = []
    X.append(X_N)
    Y.append(Y_1)
    while counter < N:
        X.append(eq_line(Y[-1]))
        Y.append(Y[-1])
        X.append(X[-1])
        Y.append(op_line(X[-1]))
        counter += 1

    return (X, Y)


def plot_VLE_XY_reg(Tin, Tout, Treg, RHin, RHout, Y_1, N):
    range = find_operating_range(Tin, Tout, Treg, RHin, RHout, .1)

    Tin, Tout, Treg = convert_F_to_C(Tin), convert_F_to_C(Tout), convert_F_to_C(Treg)
    x_range = np.linspace(range[0], range[1], 100)
    y_range = p_vap_solution(x_range, Treg) / P

    X_eq = (1 - convert_x_to_X(x_range))
    Y_eq = convert_x_to_X(y_range)

    # Creates straight line between two points
    X_RH = (X_eq[0], X_eq[99])
    y_RH = convert_RH_to_Pa(RHout, Tout) / P
    Y_RH = (convert_x_to_X(y_RH), convert_x_to_X(y_RH))

    mpl_plt.plot(X_eq, Y_eq, label = "Equilibrium Line")
    mpl_plt.plot(X_RH, Y_RH, linestyle = '--', label = "Outdoor R.H")

    mpl_plt.legend(loc='upper left')
    mpl_plt.xlabel("H2O liquid mole ratio")
    mpl_plt.ylabel("H2O vapor mole ratio")
    mpl_plt.savefig("Images/VLE_XY_reg.png")
    mpl_plt.show()


def VLE(y_in, V, T1, guess):
    x_ul = find_operating_range(72, 85, 105, .6, .9, .1)[1]

    T1 = convert_F_to_C(T1)

    L =  1/x_ul
    x_in = 1 - x_ul

    f = lambda x: y_in * V + L * x_in - (x * L * (1 - x_in) / (1 - x)) \
                     + p_vap_solution(x, T1)/P * V * (1 - y_in) / (1 - p_vap_solution(x, T1)/P)

    x = sci_opt.fsolve(f, guess)

    V_out = V*(1 - y_in) / (1 - p_vap_solution(1-x, T1)/P)

    x_abs = p_vap_solution(x, T1) / P

    return (y_in*V - x_abs*V_out) / 1.5


def p_control(kc, m_in, V_0):

    mol_water = 272.64

    mol = [mol_water * 0.9]

    V = [V_0]
    mol_air = 10280.5

    for t in range(0, 3600):
        y_in = (mol[t] + m_in) / mol_air
        mol.append(mol[t] + m_in - VLE(y_in, V[t], 72, 0.1))
        RH = mol[t+1] / 272.64
        e = 0.6 - RH
        V.append(kc * e + V_0)

    time_range = np.linspace(0, 3600, len(V))

    mpl_plt.plot(time_range, np.array(mol) / mol_water)
    mpl_plt.xlabel("Time (s)")
    mpl_plt.ylabel("RH")
    mpl_plt.savefig("Images/P_RH_{m_in}".format(m_in = m_in))
    mpl_plt.show()
    mpl_plt.close()

    mpl_plt.plot(time_range, V)
    mpl_plt.xlabel("Time (s)")
    mpl_plt.ylabel("Air Flow Rate (mol/s)")
    mpl_plt.savefig("Images/P_air_{m_in}".format(m_in = m_in))
    mpl_plt.show()


def pi_control(kc, tau_i, m_in, V_0):

    mol_water = 272.64

    mol = [mol_water * 0.9]

    V = [V_0]
    mol_air = 10280.5

    e = [0.0]

    for t in range(0, 3600):
        y_in = (mol[t] + m_in) / mol_air
        mol.append(mol[t] + m_in - VLE(y_in, V[t], 72, 0.1))
        RH = mol[t+1] / 272.64
        e.append(0.6 - RH)
        V.append(kc * (e[t+1] + sum(e)/tau_i) + V_0)

    time_range = np.linspace(0, 3600, len(V))

    mpl_plt.plot(time_range, np.array(mol) / mol_water)
    mpl_plt.xlabel("Time (s)")
    mpl_plt.ylabel("RH")
    mpl_plt.savefig("Images/PI_RH_{m_in}.png".format(m_in = m_in))
    mpl_plt.show()
    mpl_plt.close()

    mpl_plt.plot(time_range, V)
    mpl_plt.xlabel("Time (s)")
    mpl_plt.ylabel("Air Flow Rate (mol/s)")
    mpl_plt.savefig("Images/PI_air_{m_in}.png".format(m_in = m_in))
    mpl_plt.show()


def plot_power_pi():

    area = 5

    hours = [x for x in range(6, 19)]
    irradiance = [0, 0.2, 0.46, 0.7, 0.89, 1.0, 1.04, 1.0, 0.88, 0.7, 0.46, 0.2, 0]
    power = 0.2 * area * np.array(irradiance)

    mpl_plt.plot(hours, power)

    conversion_factor = 430 / 61.22 / 1000

    high_hours = [6, 7]
    high_pump_power = [50 * conversion_factor, 50 * conversion_factor]

    low_hours = [7, 18]
    low_pump_power = [45 * conversion_factor, 45 * conversion_factor]

    mpl_plt.plot(high_hours, high_pump_power)
    mpl_plt.plot(low_hours, low_pump_power)
    mpl_plt.xlabel("Time (hr)")
    mpl_plt.ylabel("Power (kW)")
    mpl_plt.savefig("images/PI_power")
    mpl_plt.show()





def pid_control(kc, tau_i, tau_d, m_in, V_0):

    mol_water = 272.64

    mol = [mol_water * 0.9]

    V = [V_0]
    mol_air = 10280.5

    e = [0.0]

    for t in range(0, 3600):
        y_in = (mol[t] + m_in) / mol_air
        mol.append(mol[t] + m_in - VLE(y_in, V[t], 72, 0.1))
        RH = mol[t+1] / 272.64
        e.append(0.6 - RH)
        V.append(kc * (e[t+1] + sum(e)/tau_i + tau_d * (e[t+1] - e[t])) + V_0)

    time_range = np.linspace(0, 3600, len(V))

    mpl_plt.plot(time_range, np.array(mol) / mol_water)
    mpl_plt.xlabel("Time (s)")
    mpl_plt.ylabel("RH")
    mpl_plt.show()
    mpl_plt.savefig("Images/PID_RH")
    mpl_plt.close()

    mpl_plt.plot(time_range, V)
    mpl_plt.xlabel("Time (s)")
    mpl_plt.ylabel("Air Flow Rate (mol/s)")
    mpl_plt.savefig("Images/PID_air")
    mpl_plt.show()

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
# plot_VLE_XY_abs(72, 85, 105, .6, .8, 0.017, 3)
# plot_VLE_XY_abs(72, 85, 105, .6, .8, 0.019, 3)
# plot_VLE_XY_abs(72, 85, 105, .6, .8, 0.021, 3)
# plot_VLE_XY_reg(72, 85, 105, .6, .8, 0.017, 1)

# print(VLE(0.04, 10, 72, 0.1))
# p_control(-3, 0.1, 15)
# pi_control(-35, 100, 0.1, 10)
# pi_control(-35, 100, 0.0, 10)
# pid_control(-35, 100, 100, 0.1, 10)

plot_power_pi()