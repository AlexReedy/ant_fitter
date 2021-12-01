from fittingLib import *
import timeit
fit = FittingLibrary(pause=2)

start = timeit.default_timer()
counter = 0
for data_set in fit.data_sets:
    ind_start = timeit.default_timer()

    fit.import_data(file=data_set)

    fit.plot_mag(save=True, show=True)
    fit.plot_flux(save=True, show=True)

    fit.sigma_clipping()
    fit.plot_sigma_clip(save=True, show=True)

    fit.get_average()
    fit.plot_sigma_clip_avg(save=True, show=True)
    fit.get_fit_parameters()

    ind_end = timeit.default_timer() - ind_start

    print(f'{data_set} [{counter} / {len(fit.data_sets)}] Completed:\n'
          f'Runtime: {int(ind_end)} sec\n'
          f'Moving to Next Data Set!')

    counter += 1

runtime = timeit.default_timer() - start

print(f'Total Runtime: {int(runtime)} sec')
