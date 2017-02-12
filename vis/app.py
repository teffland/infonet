from flask import Flask, render_template
app = Flask(__name__)

import os
import os.path as osp
import json
import yaml
from io import open

experiment_dir = '../experiments/'
@app.route('/')
def home():
    experiments =  [ d for d in os.listdir(experiment_dir)
                     if osp.isdir(osp.join(experiment_dir,d)) ]
    return render_template('home.html', experiments=experiments)

@app.route('/experiment/<experiment>')
def experiment_home(experiment):
    exp_dir = osp.join(experiment_dir, experiment)
    print 'exp dir', exp_dir
    exp_config = yaml.load(open(osp.join(exp_dir, 'config.yaml')))
    eval_stats = json.load(open(osp.join(exp_dir, 'eval_stats.json')))
    train_stats = []
    for line in open(osp.join(exp_dir, 'train_stats.json')):
        train_stats.append(json.loads(line))
    print train_stats[0].keys()
    batch_losses = [s['epoch_losses'] for s in train_stats]
    dev_scores = [s['dev_score'] for s in train_stats]
    return render_template('experiment_home.html',
                           experiment_name=experiment,
                           exp_config=exp_config,
                           eval_stats=eval_stats,
                           train_stats=train_stats,
                           batch_losses=batch_losses,
                           dev_scores=dev_scores)


##################
# Jinja2 Filters #
##################
@app.template_filter('isdict')
def is_dict(d):
    return type(d) is dict
