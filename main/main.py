from optparse import OptionParser

import torch

from algorithms.naive.random_model import rand_experiment, RandomModel
from algorithms.qsom.experiments import *
from algorithms.qsom.qsom import QSOM
from scenarios.scenario_declaration import *
from runner import Runner

string_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={string_device}")
device = torch.device(string_device)

models = {
    "QSOM": QSOM,
    "RandomModel": RandomModel,
}

scenarios = {
    "One": ScenarioOne,
    "Two": ScenarioTwo,
    "Three": ScenarioThree,
    "Four": ScenarioFour,
    "Five": ScenarioFive,
    "Six": ScenarioSix,
    "Seven": ScenarioSeven,
    "Eight": ScenarioEight,
    "Nine": ScenarioNine,
    "Ten": ScenarioTen,
    "Eleven": ScenarioEleven,
    "Twelve": ScenarioTwelve,
    "Thirteen": ScenarioThirteen,
    "Fourteen": ScenarioFourteen,
    "Fifteen": ScenarioFifteen,
    "Sixteen": ScenarioSixteen,
    "Seventeen": ScenarioSeventeen,
    "Eighteen": ScenarioEighteen,
    "TwentyThree": ScenarioTwentyThree,
    "TwentyFour": ScenarioTwentyFour,
    "TwentyFive": ScenarioTwentyFive,
    "TwentySix": ScenarioTwentySix,
    "TwentySeven": ScenarioTwentySeven,
    "TwentyEight": ScenarioTwentyEight,
    "Nineteen": ScenarioNineteen,
    "Twenty": ScenarioTwenty,
    "TwentyOne": ScenarioTwentyOne,
    "TwentyTwo": ScenarioTwentyTwo,
    "TwentyNine": ScenarioTwentyNine,
    "Thirty": ScenarioThirty,
}

experiments_qsom = {
    "qsom_1": qsom_1,
    "qsom_2": qsom_2,
    "qsom_3": qsom_3,
    "qsom_4": qsom_4,
    "qsom_5": qsom_5,
    "qsom_6": qsom_6,
    "qsom_7": qsom_7,
    "qsom_8": qsom_8,
    "qsom_9": qsom_9,
    "qsom_10": qsom_10,
    "qsom_11": qsom_11,
    "qsom_12": qsom_12,
    "qsom_13": qsom_13,
}

global_experiments = {
    "RandomModel": {"RandomModel": rand_experiment},
    "QSOM": experiments_qsom,
}


def main():
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-e", "--experiment", dest="experiment",
                      help="Experiment selected: ex: qsom_1")
    parser.add_option("-m", "--model", dest="model",
                      help="Model selected, ex: QSOM")
    parser.add_option("-s", "--scenario", dest="scenario",
                      help="Scenario selected, ex: ScenarioFive")
    parser.add_option("-t", "--mode", dest="mode",
                      help="mode of runs, ex: training, evaluation ..")
    parser.add_option("-d", "--dryrun", dest="dryrun",
                      action="store_true",
                      help="Check parameters")
    parser.add_option("-p", "--path", dest="path",
                      action="store_true",
                      help="Used for aim folder specification")
    parser.add_option("-a", "--allrun", dest="all_run",
                      action="store_true",
                      help="Check parameters")

    (options, args) = parser.parse_args()
    if len(args) != 0:
        parser.error("incorrect number of arguments")


    if not options.dryrun:
        if not options.all_run:
            experiment = global_experiments[options.model][options.experiment]
            model = models[options.model]
            scenario = scenarios[options.scenario]

            runner = Runner(hyper_parameters=experiment, model=model, device=device, scenario=scenario(),
                            mode=options.mode)
            runner.start()
        else:
            inp = input(
                'Be careful : launch all experiments at your own risk. This may take a while and many ressources.\n'
                'Type y if you want to continue, all other will interrupt.\n')
            if inp == 'y' or inp == 'Y':
                script_one(options.mode)

    else:
        experiment = global_experiments[options.model][options.experiment]
        model = models[options.model]
        scenario = scenarios[options.scenario]
        if experiment is not None and model is not None and scenario is not None:
            quit(0)
        else:
            raise 'Not fully parametrized'


def script_one(mode):
    for scenario in scenarios.values():
        print(scenario.__name__)
        for model in models.values():
            print(model.__name__)

            experiments = global_experiments[model.__name__]

            for experiment in experiments.values():
                runner = Runner(hyper_parameters=experiment, model=model, device=device, scenario=scenario(), mode=mode)
                runner.start(True)


if __name__ == '__main__':
    main()
