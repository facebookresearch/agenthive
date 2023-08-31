''' Use this script to comapare multiple results \n
    Usage: python viz_resulyts.py -j expdir1_group0 expdir2_group0 -j expdir3_group1 expdir4_group1 -k "key1" "key2"...
'''
from vtils.plotting import simple_plot
import argparse
from scipy import signal
import pandas
import glob

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
        '-lf', '--log_file', type=str, default="log.csv", help='name of log file (with extension)')
    parser.add_argument(
        '-cf', '--config_file', type=str, default="job_config.json", help='name of config file (with extension)')
    parser.add_argument(
        '-t', '--title', type=str, default=None, help='Title of the plot')
    parser.add_argument(
        '-l', '--label', action='append', nargs='?', help='job group label')
    parser.add_argument(
        '-s', '--smooth', type=int, default=21, help='window for smoothing')
    parser.add_argument(
        '-y', '--ykeys', nargs='+', default=["eval_score", 'norm_score'], help='yKeys to plot')
    parser.add_argument(
        '-x', '--xkey', default="total_num_samples", help='xKey to plot')
    parser.add_argument(
        '-i', '--index', type=int, default=-4, help='index in log filename to use as labels')
    args = parser.parse_args()

    # scan labels
    if args.label is not None:
        assert (len(args.job) == len(args.label)), "The number of labels has to be same as the number of jobs"
    else:
        args.label = [''] * len(args.job)

    # for all the algo jobs
    for ialgo, algo_dir in enumerate(args.job):
        print("algo> "+algo_dir)
        envs_dirs = glob.glob(algo_dir+"/*/")
        # for all envs inside the algo
        nenv = len(envs_dirs)
        for ienv, env_dir in enumerate(sorted(envs_dirs)):
            print("env>> "+env_dir)
            run_dirs = glob.glob(env_dir+"/*/")

            # all the seeds/ variations within the env
            for irun, run_dir in enumerate(sorted(run_dirs)):
                print("run> "+run_dir)
                title = run_dir.split('/')[3]
                title = title[:title.find('-v')]

                # for log_file in get_files(env_dir, args.file):
                log_file = get_files(run_dir, args.log_file)
                log = get_log(filename=log_file[0], format="csv")

                # validate keys
                for key in [args.xkey]+args.ykeys:
                    assert key in log.keys(), "{} not present in available keys {}".format(key, log.keys())

                nykeys =  len(args.ykeys)
                for iykey, ykey in enumerate(sorted(args.ykeys)):
                    simple_plot.plot(xdata=log[args.xkey]/1e6,
                        ydata=smooth_data(log[ykey], args.smooth),
                        legend='job_name',
                        subplot_id=(nenv, nykeys, nykeys*ienv+iykey+1),
                        xaxislabel=args.xkey+'(M)',
                        plot_name=title,
                        yaxislabel=ykey,
                        fig_size=(4*nykeys, 4*nenv),
                        fig_name='SAC performance'
                        )
    # simple_plot.show_plot()
    simple_plot.save_plot(args.job[0]+'RS-SAC.pdf')


if __name__ == '__main__':
    main()
