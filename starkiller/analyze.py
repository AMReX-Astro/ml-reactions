# Plot data from a Reactions ML surrogate model
import h5py
import imageio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from TrainingHistory import TrainingHistory

history = TrainingHistory()
history.load_history("training_history.h5")

# get numpy versions of x,y,f on the cpu for plotting
xnp = history.training_inputs
ynp = history.training_truth
fnp = history.training_truth_rhs

training_images = []

fig, (axis_p, axis_f, axis_e) = plt.subplots(nrows=3, ncols=1, figsize=(8,8), dpi=150)
axis_e1 = axis_e.twinx()
axis_p_t = axis_p.twinx()
axis_f_t = axis_f.twinx()

for i, epoch in enumerate(history.epochs):
    # clear previously drawn curves
    axis_p.clear()
    axis_p_t.clear()

    axis_p.set_ylabel('Solution', fontsize=22)

    pnp = history.model_history[i]

    for n in range(history.nspec):
        axis_p.plot(xnp, pnp[:,n],
                    color='green', lw=3, alpha=0.5)

        axis_p.scatter(xnp, ynp[:,n],
                    color='blue', alpha=0.5, s=20)
        
    axis_p_t.plot(xnp, pnp[:,history.net_itemp],
                color='green', lw=3, alpha=0.5,
                label='p(t)')

    axis_p_t.scatter(xnp, ynp[:,history.net_itemp],
                    color='red', alpha=0.5, s=20,
                    label='x(t)')

    # Plot analytic rhs vs prediction rhs
    pfnp = history.model_rhs_history[i]
    dpdxnp = history.model_grad_history[i]

    # clear previously drawn curves
    axis_f.clear()
    axis_f_t.clear()

    axis_f.set_ylabel('Gradient', fontsize=22)

    for n in range(history.nspec):
        axis_f.plot(xnp, pfnp[:,n],
                    color='green', lw=3, alpha=0.5)

        axis_f.plot(xnp, dpdxnp[:,n],
                    color='magenta', lw=3, ls=':', alpha=0.5)

        axis_f.scatter(xnp, fnp[:,n],
                    color='blue', alpha=0.5, s=20)
        
    axis_f_t.plot(xnp, pfnp[:,history.net_itemp],
                color='green', lw=3, alpha=0.5,
                label='f(p(t))')

    axis_f_t.plot(xnp, dpdxnp[:,history.net_itemp],
                color='black', lw=3, ls=':', alpha=0.5,
                label='dp(t)/dt')

    axis_f_t.scatter(xnp, fnp[:,history.net_itemp],
                    color='red', alpha=0.5, s=20,
                    label='f(x(t))')

    axis_f.tick_params(axis='both', which='major', labelsize=16)
    axis_f_t.tick_params(axis='both', which='major', labelsize=16)

    axis_f_t.legend(loc='upper right', borderpad=1, framealpha=0.5)

    # get min/max in x/y to set label positions relative to the axes
    xmin = 0
    xmax = 1
    ymin = 0
    ymax = 1

    height = np.abs(ymax - ymin)
    width = np.abs(xmax - xmin)

    axis_p.set_xlim(xmin, xmax)
    axis_p.set_ylim(ymin, ymax)

    axis_p.text(xmin, ymax + height*0.3,
            'Step = %d' % epoch, fontdict={'size': 24, 'color': 'blue'})
    axis_p.text(xmin + width*0.5, ymax + height*0.3,
            'Train Loss = %.2e' % history.losses[i],
            fontdict={'size': 24, 'color': 'blue'})
    axis_p.text(xmin + width*0.5, ymax + height*0.1,
            'Test Loss = %.2e' % history.test_losses[i],
            fontdict={'size': 24, 'color': 'orange'})

    axis_p.tick_params(axis='both', which='major', labelsize=16)
    axis_p_t.tick_params(axis='both', which='major', labelsize=16)

    # clear previously drawn curves
    axis_e.clear()
    axis_e1.clear()

    axis_e.set_xlabel('Epoch', fontsize=22)
    axis_e.set_ylabel('E(p,x)', fontsize=22)

    axis_e.scatter([epoch], [history.losses0[i]],
                color="red", alpha=0.5)
    axis_e.plot(history.epochs[:i+1], history.losses0[:i+1],
                'b-', lw=3, alpha=0.5,
                label='E(p,x) [train]')

    axis_e.scatter([epoch], [history.test_losses[i]],
                color="red", alpha=0.5)
    axis_e.plot(history.epochs[:i+1], history.test_losses[:i+1],
                'orange', lw=3, ls="--", alpha=0.5,
                label='E(p,x) [test]')

    axis_e1.set_ylabel('E(dp/dt, f(x))', fontsize=22)

    axis_e1.scatter([epoch], [history.losses1[i]],
                color="red", alpha=0.5)
    axis_e1.plot(history.epochs[:i+1], history.losses1[:i+1],
                'g-', lw=3, alpha=0.5,
                label='E(dp/dt, f(x)) [train]')

    axis_e.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: "{:0.1f}".format(x)))

    axis_e1.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: "{:0.1f}".format(x)))

    axis_e.tick_params(axis='both', which='major', labelsize=16)
    axis_e1.tick_params(axis='both', which='major', labelsize=16)

    axis_e.legend(loc='upper right', borderpad=1, framealpha=0.5)
    axis_e1.legend(loc='upper center', borderpad=1, framealpha=0.5)

    # Draw on canvas and save image in sequence
    fig.canvas.draw()
    plt.tight_layout()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    training_images.append(image)

imageio.mimsave('./starkiller.gif', training_images, fps=20)

print("final test sample error: ", history.test_losses[-1])

prediction_test_np = history.test_model_history[-1]
y_test_np = history.test_truth

def plot_prediction_truth(label, p, t):
    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(t, p)
    ax.plot(t,t,'r')
    ax.set_xlabel("truth {}".format(label))
    ax.set_ylabel("prediction {}".format(label))
    plt.savefig("prediction_map_{}.png".format(label), dpi=300)

for n in range(history.nspec+2):
    plot_prediction_truth(n, prediction_test_np[:,n], y_test_np[:,n])

# plot the truth solution & rhs
def plot_truth(xnp, ynp, fnp):
    # plot the truth solution
    fig, axis = plt.subplots(figsize=(5,5), dpi=150)
    axis_t = axis.twinx()

    for n in range(history.nspec):
        axis.scatter(xnp, ynp[:,n],
                    color='blue', alpha=0.5)
        
    axis_t.scatter(xnp, ynp[:,history.net_itemp],
                color='red', alpha=0.5)

    axis.set_ylabel("X")
    axis.set_xlabel("t")
    axis_t.set_ylabel("T")

    plt.savefig("system_truth_sol.png", dpi=300)

    # plot the truth rhs
    fig, axis = plt.subplots(figsize=(5,5), dpi=150)
    axis_t = axis.twinx()

    for n in range(history.nspec):
        axis.scatter(xnp, fnp[:,n],
                    color='blue', alpha=0.5)
        
    axis_t.scatter(xnp, fnp[:,history.net_itemp],
                color='red', alpha=0.5)

    axis.set_ylabel("dX/dt")
    axis.set_xlabel("t")
    axis_t.set_ylabel("dT/dt")

    plt.savefig("system_truth_rhs.png", dpi=300)

plot_truth(xnp, ynp, fnp)