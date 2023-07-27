import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import trapezoid
from matplotlib import pyplot as plt
import materialdatabase as mdb
import leapythontoolbox as lpt

from dtos import *




def create_sinusoidal_flux(b_max: float):
    """
    Creates a sinusoidal waveform. The waveform has 1024 points, according to the given structure
    used in the measurements by the 'magnet challenge' of princeton university
    :param b_max: amplitude of the sinus
    :return: amplitude vector with 1024 values
    """
    time_vec = np.linspace(0, 2*np.pi, 1024)
    amplitude_vec = b_max * np.sin(time_vec)

    return amplitude_vec


def material_dict() -> dict[str, MaterialParameters]:
    """
    Material dictionary containing the parameters in MaterialParameters-class.
    The values are taken from dissertation erika stenglein.
    :return: dict of MaterialParameters

    # page 105: formula for w_hyst
    """
    material_dict = {}

    material_dict["N87"] = MaterialParameters(param_k=1.138e-5, param_gamma=0.7683, frequency_max=500e3, b_sat=0.49)
    material_dict["N49"] = MaterialParameters(param_k=4.344e-6, param_gamma=0.7995, frequency_max=500e3, b_sat=0.49)
    material_dict["3F3"] = MaterialParameters(param_k=1.918e-5, param_gamma=0.7340, frequency_max=500e3, b_sat=0.44)
    material_dict["3C90"] = MaterialParameters(param_k=7.425e-5, param_gamma=0.6800, frequency_max=500e3, b_sat=0.47)
    material_dict["3C97"] = MaterialParameters(param_k=1.831e-5, param_gamma=0.7820, frequency_max=200e3, b_sat=0.53)

    return material_dict


def w_hyst_polynom_dict() -> dict[str, Polynomials]:
    polynom_dict = {}
    polynom_dict["N87"] = Polynomials(a_1=-5.46e-1, a_2=7.49, a_3=5.68e-1, a_4=0, b_1=8.11, b_2=6.45e-1, beta=2.81)
    polynom_dict["N49"] = Polynomials(a_1=1.68e-1, a_2=-2.53e-1, a_3=1.29e1, a_4=-1.66, b_1=6.79, b_2=6.35e-1, beta=2.44)
    polynom_dict["3F3"] = Polynomials(a_1=-5.39e-1, a_2=6.62, a_3=8.62e-1, a_4=0, b_1=6.69, b_2=6.85e-1, beta=2.4)
    polynom_dict["3C90"] = Polynomials(a_1=2.91e-1, a_2=1.45, a_3=4.24, a_4=-8.32e-1, b_1=1.04e1, b_2=3.75e-1, beta=2.73)
    polynom_dict["3C97"] = Polynomials(a_1=5.98e-2, a_2=4.48e-1, a_3=6.6, a_4=-1.84, b_1=7.16, b_2=9.3e-1, beta=2.71)

    return polynom_dict


def w_hyst(material_name: str, b_waveform_frequency_independent_1024: float, dc_flux: float = 0):
    """
    Calculates the w_hyst energy, according to formula 4.19 page 105 (dissertation stenglein).

    :param material_name: material name
    :param b_waveform_frequency_independent_1024:
    :param dc_flux: DC-flux offset. Set to zero by default
    :return:
    """

    delta_flux = np.max(b_waveform_frequency_independent_1024) - np.min(b_waveform_frequency_independent_1024)

    material_polynomials = w_hyst_polynom_dict()[material_name]
    material_parameters = material_dict()[material_name]

    w_hyst_material = (material_polynomials.a_1 * (delta_flux / material_parameters.b_sat) +
                       material_polynomials.a_2 * (delta_flux / material_parameters.b_sat) ** 2 +
                       material_polynomials.a_3 * (delta_flux / material_parameters.b_sat) ** 3 +
                       material_polynomials.a_4 * (delta_flux / material_parameters.b_sat) ** 4) * \
                      (1 + material_polynomials.b_1 * np.abs(dc_flux / material_parameters.b_sat) ** material_polynomials.beta * (1 - material_polynomials.b_2 * delta_flux / material_parameters.b_sat))

    return w_hyst_material

def f_rate(material_name:str, b_waveform_frequency_independent_1024, frequency: float):


    material_factors = material_dict()[material_name]

    delta_b = np.max(b_waveform_frequency_independent_1024) - np.min(b_waveform_frequency_independent_1024)

    time_s = np.linspace(0, 1/frequency, 1024)

    # derivation
    # according to https://im-coder.com/zweite-ableitung-in-python-scipy-numpy-pandas.html
    fitted_function = UnivariateSpline(time_s, b_waveform_frequency_independent_1024, s=0, k=4)
    #plt.plot(time_s, b_waveform_frequency_independent_1024, 'ro', label='original data')
    #plt.plot(time_s, fitted_function(time_s), label="fitted function")
    amplitude_2nd_derivation = fitted_function.derivative(n=2)
    #plt.plot(time_s, amplitude_2nd_derivation(time_s), label="amplitude 2nd derivation")
    integrated_function = trapezoid(np.abs(amplitude_2nd_derivation(time_s)), time_s)
    #plt.legend()
    #plt.grid()
    # plt.show()

    integral_part = integrated_function / delta_b

    effective_frequency = frequency * (1 + material_factors.param_k * integral_part ** material_factors.param_gamma)

    return effective_frequency


def comparison_plots_datasheet_f_p(material_name: str):
    material_database = mdb.MaterialDatabase()
    material_data = material_database.load_database()[material_name]

    relative_core_loss_frequency_list = material_data["manufacturer_datasheet"]["relative_core_loss_frequency"]

    for count, relative_core_loss_frequency in enumerate(relative_core_loss_frequency_list):

        temperature = relative_core_loss_frequency["temperature"]
        flux_density = relative_core_loss_frequency["flux_density"]

        if temperature == 25:
            # plot the datasheet data
            plt.loglog(relative_core_loss_frequency["frequency"], relative_core_loss_frequency["power_loss"], label=f"B = {int(flux_density * 1000)} mT, T = {temperature} °C", color=lpt.gnome_colors_list[count])

            # calculate an plot the stenglein data
            frequency_vec = np.linspace(30e3, 500e3)
            b_waveform_frequency_independent_1024 = create_sinusoidal_flux(flux_density)
            p_hyst_stenglein = calculate_core_loss_f_p(material_name, b_waveform_frequency_independent_1024, frequency_vec)
            plt.loglog(frequency_vec, p_hyst_stenglein,"--", color=lpt.gnome_colors_list[count])


    measurement_stenglein_200mt = np.genfromtxt('stenglein_measurement_n87/400mt_25c.csv', delimiter=';', )
    plt.loglog(measurement_stenglein_200mt[:,0], measurement_stenglein_200mt[:,1], '*', label='measurement stenglein 200 mT', color="orange")
    measurement_stenglein_100mt = np.genfromtxt('stenglein_measurement_n87/200mt_25c.csv', delimiter=';', )
    plt.loglog(measurement_stenglein_100mt[:, 0], measurement_stenglein_100mt[:, 1], '*', label='measurement stenglein 100 mT', color="green")


    plt.grid(which="both")
    plt.title(material_name)
    plt.xlabel(f"Frequency in Hz")
    plt.ylabel(f"Power loss density in kW/m³")
    plt.legend()
    plt.show()


def comparison_plots_datasheet_b_p(material_name: str):
    material_database = mdb.MaterialDatabase()
    material_data = material_database.load_database()[material_name]

    relative_core_loss_flux_density_list = material_data["manufacturer_datasheet"]["relative_core_loss_flux_density"]

    for count, relative_core_loss_flux_density in enumerate(relative_core_loss_flux_density_list):
        temperature = relative_core_loss_flux_density["temperature"]
        frequency = relative_core_loss_flux_density["frequency"]

        if temperature == 25:
            # plot the datasheet data
            plt.semilogy(relative_core_loss_flux_density["flux_density"], relative_core_loss_flux_density["power_loss"],
                       label=f"f = {frequency} Hz, T = {temperature} °C", color=lpt.gnome_colors_list[count])

            # calculate an plot the stenglein data
            p_hyst_stenglein = calculate_core_loss_b_p(material_name, frequency, relative_core_loss_flux_density["flux_density"])
            plt.semilogy(relative_core_loss_flux_density["flux_density"], p_hyst_stenglein, "--", color=lpt.gnome_colors_list[count])

    plt.grid(which="both")
    plt.title(material_name)
    plt.xlabel(f"Flux density in T")
    plt.ylabel(f"Power loss density in kW/m³")
    plt.legend()
    plt.show()

def calculate_core_loss_f_p(material_name: str, b_waveform_frequency_independent_1024, frequency_vec):
    """
    Calculate the core losses for certain frequencies of the same flux-density waveform

    :param material_name: material name
    :param b_waveform_frequency_independent_1024: waveform in 1024 steps, time independent
    :param frequency_vec: vector with frequencies to be calculated, e.g. [10e3, 100e3, 1e6]
    :return: loss vector for the given frequencies
    """

    # calculate the frequency independent quasistatic hysteresis losses
    quasistatic_hyst_losses = w_hyst(material_name, b_waveform_frequency_independent_1024=b_waveform_frequency_independent_1024)

    # calculate the frequency dependent part by using the effective frequency
    p_hyst_vec = []
    for frequency in frequency_vec:
        f_eff = f_rate(material_name, b_waveform_frequency_independent_1024, frequency)

        p_hyst = quasistatic_hyst_losses * f_eff
        p_hyst_vec.append(p_hyst)

    return p_hyst_vec


def calculate_core_loss_b_p(material_name:str, frequency: float, b_vec_sinusoidal):
    p_hyst_vec = []
    # calculate the frequency independent quasistatic hysteresis losses
    for flux_vec in b_vec_sinusoidal:
        b_waveform_frequency_independent_1024 = create_sinusoidal_flux(flux_vec)
        quasistatic_hyst_losses = w_hyst(material_name, b_waveform_frequency_independent_1024=b_waveform_frequency_independent_1024)

        f_eff = f_rate(material_name, b_waveform_frequency_independent_1024, frequency)

        p_hyst = quasistatic_hyst_losses * f_eff
        p_hyst_vec.append(p_hyst)

    return p_hyst_vec




if __name__ == "__main__":
    comparison_plots_datasheet_f_p("N87")
    #comparison_plots_datasheet_f_p("N49")
    #comparison_plots_datasheet_b_p("N87")

    # b_waveform = create_sinusoidial_flux(b_max=0.1)
    # material_name = "N87"
    # frequency_vec = np.linspace(30e3, 500e3)
    #
    # quasistatic_hyst_losses = w_hyst(material_name, b_waveform_frequency_independent_1024=b_waveform)
    #
    #
    # p_hyst_vec = []
    #
    # for frequency in frequency_vec:
    #     f_eff = f_rate(material_name, b_waveform, frequency)
    #
    #     p_hyst = quasistatic_hyst_losses * f_eff
    #     p_hyst_vec.append(p_hyst)
    #
    # print(f"{quasistatic_hyst_losses = }")
    # print(f"{f_eff = }")
    # print(f"{p_hyst = }")
    # plt.loglog(frequency_vec, p_hyst_vec)
    # plt.grid()
    # plt.show()



