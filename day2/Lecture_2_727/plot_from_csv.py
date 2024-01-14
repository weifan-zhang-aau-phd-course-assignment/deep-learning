import Lecture_2_727
import matplotlib.pyplot as plt

runner_sgd = Lecture_2_727.model_runner("SGD", None, None, None)
runner_adam = Lecture_2_727.model_runner("Adam", None, None, None)

runner_sgd.read_csv()
runner_adam.read_csv()

plt.rcParams.update({'font.size': 14})

fig = plt.figure(figsize=(16, 18))
Lecture_2_727.plot_model(fig, 1, 2, runner_sgd)
Lecture_2_727.plot_model(fig, 2, 2, runner_adam)
fig.subplots_adjust(hspace=0.5)
plt.show()
