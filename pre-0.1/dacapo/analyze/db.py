from pymongo import MongoClient
import configargparse

parser = configargparse.ArgParser(
    default_config_files=['~/.config/dacapo', './dacapo.conf'])
parser.add(
    '-c', '--config',
    is_config_file=True,
    help="The config file to use.")
parser.add(
    '-m', '--mongo_db_host',
    help="The mongo DB host to use.")
parser.add(
    '-f', '--file',
    help="Needed to work within jupyter")

def read_runs():

    options = parser.parse()
    client = MongoClient(options.mongo_db_host)

    run_docs = list(client['dacapo_v01'].runs.find())
    run_docs = run_docs[-10:]
    runs = []

    print("Reading runs...")
    for run_doc in run_docs:

        train_docs = client['dacapo_v01'].train.find({
            'run': run_doc['id'],
            'repetition': run_doc['repetition']
        })
        train_docs = list(train_docs)
        train_losses = [t['loss'] for t in train_docs]
        train_iterations = [t['iteration'] for t in train_docs]

        validation_docs = client['dacapo_v01'].validate.find({
            'run': run_doc['id'],
            'repetition': run_doc['repetition']
        })
        validation_docs = list(validation_docs)
        validation_iterations = [v['iteration'] for v in validation_docs]
        score_names = validation_docs[0]['sample_0'].keys()
        validation_scores = {}
        for score_name in score_names:
            validation_scores[s] = []
        for v in validation_docs:
            for score_name in score_names:
                validation_scores[s].append(
                    [v[s][score_name]
                    for s in v if s.startswith('sample_')])

        runs.append({
            'train_losses': [train_iterations, train_losses],
            'validation_scores': [validation_iterations, validation_scores],
            'task': run_doc['task'],
            'model': run_doc['model'],
            'optimizer': run_doc['optimizer'],
            'id': run_doc['id'],
            'repetition': run_doc['repetition']
        })
    print("...done.")

    return runs
