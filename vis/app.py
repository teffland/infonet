from flask import Flask, render_template
app = Flask(__name__)

import os
import os.path as osp
import json
import yaml
from io import open

experiment_dir = '../experiments/'
#################
# URL functions #
#################
@app.route('/')
def home():
    experiments =  get_valid_experiments()
    return render_template('home.html', experiments=experiments)

@app.route('/experiment/<experiment>')
def experiment_home(experiment):
    context = {'experiment_name':experiment}
    # load in the config, eval stats, and training stats
    exp_dir = osp.join(experiment_dir, experiment)
    context['exp_config'] = yaml.load(open(osp.join(exp_dir, 'config.yaml')))
    context['eval_stats'] = json.load(open(osp.join(exp_dir, 'eval_stats.json')))
    train_stats = []
    for line in open(osp.join(exp_dir, 'train_stats.json')):
        train_stats.append(json.loads(line))
    # print train_stats[0].keys()

    # reformat the training stats
    context['batch_losses'] = [s['epoch_losses'] for s in train_stats]
    context['dev_scores'] = [s['dev_score'] for s in train_stats]


    return render_template('experiment_home.html', **context)

@app.route('/experiment/<experiment>/preds')
def experiment_preds_home(experiment):
    doc_dir = osp.join(experiment_dir,experiment,'docs')
    docs = os.listdir(doc_dir)
    doc_names = [ doc[:-10] for doc in docs ]
    return render_template('docs_home.html',
                           experiment=experiment,
                           doc_names=doc_names)

@app.route('/experiment/<experiment>/<doc_name>')
def doc_compare(experiment, doc_name):
    fname = osp.join(experiment_dir, experiment, 'docs', doc_name)
    preds = json.load(open(fname+'_pred.yaat', encoding='utf8'))
    trues = json.load(open(fname+'_true.yaat', encoding='utf8'))
    return render_template('docs_compare.html',
                           preds=preds,
                           trues=trues)

####################
# Helper Functions #
####################
def get_valid_experiments():
    experiments =  []
    for d in os.listdir(experiment_dir):
        exp_dir = osp.join(experiment_dir,d)
        if osp.isdir(exp_dir): # is a directory
            if (osp.exists(osp.join(exp_dir, 'train_stats.json'))
            and osp.exists(osp.join(exp_dir, 'eval_stats.json'))
            and osp.exists(osp.join(exp_dir, 'config.yaml'))
            and osp.exists(osp.join(exp_dir, 'docs'))
            and osp.isdir(osp.join(exp_dir, 'docs'))):
                experiments.append(d)
    return experiments

##################
# Jinja2 Filters #
##################
@app.template_filter('isdict')
def is_dict(d):
    return type(d) is dict
