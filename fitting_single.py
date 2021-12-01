from fittingLib import *
fit = FittingLibrary(pause=.5, offset_prct=20, sigma_coefficient=5)


ant_confirm_id = '1118060051368.dat'
ant_test_id = '1118060050249.dat'

fit.import_data(file=ant_confirm_id)
fit.plot_raw()

fit.plot_mag()
fit.plot_flux()

fit.polyfit_sigma_clipping()
fit.plot_sigma_clip()

fit.get_average()
fit.plot_avg()

fit.get_fit_parameters()
fit.plot_fit_parameters()

'''
fit.plot_mag(save=True, show=False)
fit.plot_flux(save=True, show=False)

fit.sigma_clipping()
fit.plot_sigma_clip(save=True, show=False)

fit.get_average()
fit.plot_sigma_clip_avg(save=True, show=False)

fit.get_fit_parameters()
fit.plot_fitting_parameters(save=True, show=False)
'''