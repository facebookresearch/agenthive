''' Use this script to comapare multiple results \n
    Usage: python agents/NPG/plot_all_npg.py -j agents/v0.1/kitchen/NPG/outputs_kitchenJ5c_3.8/ -j agents/v0.1/kitchen/NPG/outputs_kitchenJ5d_3.9/ -j /Users/vikashplus/Projects/mj_envs/kitchen/outputs_kitchenJ8a/ -l 'v0.1(fixed_init)' -l 'v0.1(random_init)' -l 'v0.2(random_init)' -pt True
'''
from vtils.plotting import simple_plot
import argparse
from scipy import signal
import pandas
import glob
import numpy as np
import os

def get_files(search_path, file_name):
    search_path = search_path[:-1] if search_path.endswith('/') else search_path
    search_path = search_path+"*/**/"+file_name
    filenames = glob.glob(search_path, recursive=True)
    assert (len(filenames) > 0), "No file found at: {}".format(search_path)
    return filenames

# Another example, Python 3.5+
def get_files_p35(search_path, file_name):
    from pathlib import Path
    filenames = []
    for path in Path(search_path).rglob(file_name):
        filenames.append(path)
    return filenames


def get_log(filename, format="csv"):
    try:
        if format=="csv":
            data = pandas.read_csv(filename)
        elif format=="json":
            data = pandas.read_json(filename)
    except Exception as e:
        print("WARNING: Can't read %s." % filename)
        quit()
    return data

def smooth_data(y, window_length=101, polyorder=3):
    window_length = min(int(len(y) / 2),
                        window_length)  # set maximum valid window length
    # if window not off
    if window_length % 2 == 0:
        window_length = window_length + 1
    try:
        return signal.savgol_filter(y, window_length, polyorder)
    except Exception as e:
        return y # nans


# MAIN =========================================================
def main():

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-j', '--job', required=True, action='append', nargs='?', help='job group')
    parser.add_argument(
        '-lf', '--run_log', type=str, default="log.csv", help='name of log file (with extension)')
    parser.add_argument(
        '-cf', '--config_file', type=str, default="job_config.json", help='name of config file (with extension)')
    parser.add_argument(
        '-t', '--title', type=str, default=None, help='Title of the plot')
    parser.add_argument(
        '-l', '--label', action='append', nargs='?', help='job group label')
    parser.add_argument(
        '-s', '--smooth', type=int, default=21, help='window for smoothing')
    parser.add_argument(
        '-y', '--ykeys', nargs='+', default=['success_percentage', 'rwd_sparse', 'rwd_dense'], help='yKeys to plot')
    parser.add_argument(
        '-x', '--xkey', default="num_samples", help='xKey to plot')
    parser.add_argument(
        '-ei', '--env_index', type=int, default=-2, help='index in log filename to use as labels')
    parser.add_argument(
        '-pt', '--plot_train', type=bool, default=False, help='plot train perf')
    parser.add_argument(
        '-od', '--output_dir', type=str, default=None, help='Save outputs here')
    args = parser.parse_args()

    # init
    nykeys = len(args.ykeys)
    njob = len(args.job)
    nenv = -1
    env_labels = []

    # scan labels
    if args.label is not None:
        assert (njob == len(args.label)), "The number of labels has to be same as the number of jobs"
    else:
        args.label = [''] * njob

    # for all the jobs
    for ijob, job_dir in enumerate(args.job):
        print("Job> "+job_dir)
        envs_dirs = glob.glob(job_dir+"/*/")
        if nenv ==-1:
            nenv = len(envs_dirs)
        else:
            assert nenv == len(envs_dirs), f"Number of envs changed {envs_dirs}"
        for env_dir in sorted(envs_dirs):
            env_labels.append(env_dir.split('/')[args.env_index])

        # for all envs inside the exp
        env_means = []
        env_stds = []
        for ienv, env_dir in enumerate(sorted(envs_dirs)):
            print("  env> "+env_dir)

            # all the seeds/ variations runs within the env
            yruns = []
            xruns = [] # known bug: Logs will different lengths will cause a bug. Its hacked via using [:len(xdata)]
            for irun, run_log in enumerate(sorted(get_files(env_dir, args.run_log))):
                print("    run> "+run_log, flush=True)
                log = get_log(filename=run_log, format="csv")

                # validate keys
                for key in [args.xkey]+args.ykeys:
                    assert key in log.keys(), "{} not present in available keys {}".format(key, log.keys())
                if 'sample' in args.xkey: #special keys
                    xdata = np.cumsum(log[args.xkey])/1e6
                    plot_xkey = args.xkey+"(M)"
                else:
                    xdata = log[args.xkey]
                    plot_xkey = args.xkey
                yruns.append(log[args.ykeys])
                # print(xdata.shape, log[args.ykeys].shape)
                del log

            # stats over keys
            yruns = pandas.concat(yruns)
            yruns_stacked = yruns.groupby(yruns.index)
            yruns_mean = yruns_stacked.mean()
            yruns_min = yruns_stacked.min()
            yruns_max = yruns_stacked.max()
            yruns_std = yruns_stacked.std()

            # stats over jobs
            env_means.append(yruns_mean.tail(1))
            env_stds.append(yruns_std.tail(1))

            if args.plot_train:
                for iykey, ykey in enumerate(sorted(args.ykeys)):
                    h_figp,_,_= simple_plot.plot(xdata=xdata,
                            ydata=smooth_data(yruns_mean[ykey][:len(xdata)], args.smooth),
                            errmin=yruns_min[ykey][:len(xdata)],
                            errmax=yruns_max[ykey][:len(xdata)],
                            legend=args.label[ijob],
                            subplot_id=(nenv, nykeys, nykeys*ienv+iykey+1),
                            xaxislabel=plot_xkey,
                            plot_name=env_labels[ienv],
                            yaxislabel=ykey,
                            fig_size=(4*nykeys, 3*nenv),
                            fig_name='NPG performance',
                            )

        env_means = pandas.concat(env_means)
        env_stds = pandas.concat(env_stds)
        width = 1/(njob+1)

        for iykey, ykey in enumerate(sorted(args.ykeys)):
            h_figb, h_axisb, h_bar = simple_plot.bar(
                xdata=np.arange(nenv)+width*ijob,
                ydata=env_means[ykey],
                errdata=env_stds[ykey],
                width=width,
                subplot_id=(nykeys, 1, iykey+1),
                fig_size=(2+0.2*nenv, 4*nykeys),
                fig_name="Env perfs",
                yaxislabel=ykey,
                legend=args.label[ijob],
                xticklabels=env_labels[:nenv],
                # plot_name="Performance using 5M samples"
                )

    args.output_dir = args.job[-1] if args.output_dir == None else args.output_dir
    if args.plot_train:
        simple_plot.save_plot(os.path.join(args.output_dir, 'TrainPerf-NPG.pdf'), h_figp)
    simple_plot.save_plot(os.path.join(args.output_dir,'FinalPerf-NPG.pdf'), h_figb)


if __name__ == '__main__':
    main()
