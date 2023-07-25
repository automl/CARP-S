from __future__ import annotations

from yahpo_gym import BenchmarkSet, list_scenarios

from ConfigSpace import ConfigurationSpace
from smac.runhistory.dataclasses import TrialInfo

from smacbenchmarking.benchmarks.problem import SingleObjectiveProblem

# TODO write into genererate problems, how to collect all the benchmark problems automatically.
class YahpoProblem(SingleObjectiveProblem):
    COMBIS = {'lcbench': ['3945', '7593', '34539', '126025', '126026', '126029', '146212', '167104',
                          '167149', '167152', '167161', '167168', '167181', '167184', '167185',
                          '167190', '167200', '167201', '168329', '168330', '168331', '168335',
                          '168868', '168908', '168910', '189354', '189862', '189865', '189866',
                          '189873', '189905', '189906', '189908', '189909'],
              'fcnet': ['fcnet_naval_propulsion', 'fcnet_protein_structure',
                        'fcnet_slice_localization', 'fcnet_parkinsons_telemonitoring'],
              'nb301': ['CIFAR10'],
              'rbv2_svm': ['40981', '4134', '1220', '40978', '40966', '40536', '41156', '458',
                           '41157', '40975', '40994', '1468', '6332', '40670', '151', '1475',
                           '1476', '1478', '1479', '41212', '1480', '1053', '1067', '1056', '12',
                           '1487', '1068', '32', '470', '312', '38', '40982', '50', '41216', '307',
                           '40498', '181', '1464', '41164', '16', '1461', '41162', '6', '14',
                           '1494', '54', '375', '1590', '23', '41163', '1111', '41027', '40668',
                           '41138', '4135', '4538', '40496', '4534', '40900', '1457', '11', '1462',
                           '41142', '40701', '29', '37', '23381', '188', '41143', '1063', '3', '18',
                           '40979', '22', '1515', '334', '24', '1493', '28', '1050', '1049',
                           '40984', '40685', '42', '44', '46', '1040', '41146', '377', '40499',
                           '1497', '60', '40983', '4154', '469', '31', '41278', '1489', '1501',
                           '15', '300', '1485', '1486', '1510', '182', '41169'],
              'rbv2_ranger': ['4135', '40981', '4134', '1220', '4154', '4538', '40978', '375',
                              '40496', '40966', '4534', '40900', '40536', '41156', '1590', '1457',
                              '458', '469', '41157', '11', '1461', '1462', '1464', '15', '40975',
                              '41142', '40701', '40994', '23', '1468', '40668', '29', '31', '6332',
                              '37', '40670', '23381', '151', '188', '41164', '1475', '1476', '1478',
                              '1479', '41212', '1480', '41143', '1053', '41027', '1067', '1063',
                              '3', '6', '1485', '1056', '12', '14', '16', '18', '40979', '22',
                              '1515', '334', '24', '1486', '41278', '28', '1487', '1068', '1050',
                              '1049', '32', '1489', '470', '1494', '182', '312', '40984', '1501',
                              '40685', '38', '42', '44', '46', '40982', '1040', '41146', '377',
                              '40499', '50', '54', '41216', '307', '1497', '60', '1510', '40983',
                              '40498', '181', '41138', '41163', '1111', '41159', '300', '41162',
                              '23517', '41165', '4541', '41161', '41166', '40927', '41150', '23512',
                              '41168', '1493', '40996', '554', '40923', '41169'],
              'rbv2_rpart': ['41138', '4135', '40981', '4134', '40927', '1220', '4154', '40923',
                             '41163', '40996', '4538', '40978', '375', '1111', '40496', '40966',
                             '41150', '4534', '40900', '40536', '41156', '1590', '1457', '458',
                             '469', '41157', '11', '1461', '1462', '1464', '15', '40975', '41142',
                             '40701', '40994', '23', '1468', '40668', '29', '31', '6332', '37',
                             '4541', '40670', '23381', '151', '188', '41164', '1475', '1476',
                             '41159', '1478', '41169', '23512', '1479', '41212', '1480', '300',
                             '41168', '41143', '1053', '41027', '1067', '1063', '41162', '3', '6',
                             '1485', '1056', '12', '14', '16', '18', '40979', '22', '1515', '554',
                             '334', '24', '1486', '23517', '1493', '28', '1487', '1068', '1050',
                             '1049', '32', '1489', '470', '1494', '41161', '41165', '182', '312',
                             '40984', '1501', '40685', '38', '42', '44', '46', '40982', '1040',
                             '41146', '377', '40499', '50', '54', '41166', '307', '1497', '60',
                             '1510', '40983', '40498', '181'],
              'rbv2_glmnet': ['41138', '4135', '40981', '4134', '1220', '4154', '41163', '4538',
                              '40978', '375', '1111', '40496', '40966', '41150', '4534', '40900',
                              '40536', '41156', '1590', '1457', '458', '469', '41157', '11', '1461',
                              '1462', '1464', '15', '40975', '41142', '40701', '40994', '23',
                              '1468', '40668', '29', '31', '6332', '37', '4541', '40670', '23381',
                              '151', '188', '41164', '1475', '1476', '41159', '1478', '41169',
                              '23512', '1479', '41212', '1480', '300', '41168', '41143', '1053',
                              '41027', '1067', '1063', '41162', '3', '6', '1485', '1056', '12',
                              '14', '16', '18', '40979', '22', '1515', '334', '24', '1486', '23517',
                              '41278', '1493', '28', '1487', '1068', '1050', '1049', '32', '1489',
                              '470', '1494', '41161', '182', '312', '40984', '1501', '40685', '38',
                              '42', '44', '46', '40982', '1040', '41146', '377', '40499', '50',
                              '54', '41216', '41166', '307', '1497', '60', '1510', '40983', '40498',
                              '181', '554'],
              'rbv2_xgboost': ['16', '40923', '41143', '470', '1487', '40499', '40966', '41164',
                               '1497', '40975', '1461', '41278', '11', '54', '300', '40984', '31',
                               '1067', '1590', '40983', '41163', '41165', '182', '1220', '41159',
                               '41169', '42', '188', '1457', '1480', '6332', '181', '1479', '40670',
                               '40536', '41138', '41166', '6', '14', '29', '458', '1056', '1462',
                               '1494', '40701', '12', '1493', '44', '307', '334', '40982', '41142',
                               '38', '1050', '469', '23381', '41157', '15', '4541', '23', '4134',
                               '40927', '40981', '41156', '3', '1049', '40900', '1063', '23512',
                               '40979', '1040', '1068', '41161', '22', '1489', '41027', '24',
                               '4135', '23517', '1053', '1468', '312', '377', '1515', '18', '1476',
                               '1510', '41162', '28', '375', '1464', '40685', '40996', '41146',
                               '41216', '40668', '41212', '32', '60', '4538', '40496', '41150',
                               '37', '46', '554', '1475', '1485', '1501', '1111', '4534', '41168',
                               '151', '4154', '40978', '40994', '50', '1478', '1486', '40498'],
              'rbv2_aknn': ['41138', '40981', '4134', '40927', '1220', '4154', '41163', '40996',
                            '4538', '40978', '375', '1111', '40496', '40966', '41150', '4534',
                            '40900', '40536', '41156', '1590', '1457', '458', '469', '41157', '11',
                            '1461', '1462', '1464', '15', '40975', '41142', '40701', '40994', '23',
                            '1468', '40668', '29', '31', '6332', '37', '4541', '40670', '23381',
                            '151', '188', '41164', '1475', '1476', '41159', '1478', '41169',
                            '23512', '1479', '41212', '1480', '300', '41168', '41143', '1053',
                            '41027', '1067', '1063', '41162', '3', '6', '1485', '1056', '12', '14',
                            '16', '18', '40979', '22', '1515', '554', '334', '24', '1486', '23517',
                            '41278', '1493', '28', '1487', '1068', '1050', '1049', '32', '1489',
                            '470', '1494', '41161', '41165', '182', '312', '40984', '1501', '40685',
                            '38', '42', '44', '46', '40982', '1040', '41146', '377', '40499', '50',
                            '54', '41216', '41166', '307', '1497', '60', '1510', '40983', '40498',
                            '181', '40923'],
              'rbv2_super': ['41138', '40981', '4134', '1220', '4154', '41163', '4538', '40978',
                             '375', '1111', '40496', '40966', '4534', '40900', '40536', '41156',
                             '1590', '1457', '458', '469', '41157', '11', '1461', '1462', '1464',
                             '15', '40975', '41142', '40701', '40994', '23', '1468', '40668', '29',
                             '31', '6332', '37', '40670', '23381', '151', '188', '41164', '1475',
                             '1476', '1478', '41169', '1479', '41212', '1480', '300', '41143',
                             '1053', '41027', '1067', '1063', '41162', '3', '6', '1485', '1056',
                             '12', '14', '16', '18', '40979', '22', '1515', '334', '24', '1486',
                             '1493', '28', '1487', '1068', '1050', '1049', '32', '1489', '470',
                             '1494', '182', '312', '40984', '1501', '40685', '38', '42', '44', '46',
                             '40982', '1040', '41146', '377', '40499', '50', '54', '307', '1497',
                             '60', '1510', '40983', '40498', '181'],
              'iaml_ranger': ['40981', '41146', '1489', '1067'],
              'iaml_rpart': ['40981', '41146', '1489', '1067'],
              'iaml_glmnet': ['40981', '41146', '1489', '1067'],
              'iaml_xgboost': ['40981', '41146', '1489', '1067'],
              'iaml_super': ['40981', '41146', '1489', '1067']}

    def __init__(self, bench: str, instance: str, budget_type: str, metric,  lower_is_better:
    bool =
    True):
        """Initialize a Yahpo problem.

        Parameters
        ----------
        scenario : str Scenario name.
        instance : str Instance name.
        budget_type : str Budget type that is available in the instance
        metric : str Metric to optimize for (depends on the Benchmark instance e.g. lcbench)
        """
        super().__init__()

        assert bench in list_scenarios(), f'The scenario you choose is not available.'
        assert str(instance) in self.COMBIS[
            bench], f'The instance you choose is not available in ' \
                       f'{bench}.'

        self.scenario = bench
        self.instance = str(instance)

        self._problem = BenchmarkSet(scenario=bench, instance=self.instance)
        self._configspace = self._problem.get_opt_space(drop_fidelity_params=True)
        self.fidelity_space = self._problem.get_fidelity_space()
        self.fidelity_dims = list(self._problem.get_fidelity_space()._hyperparameters.keys())

        self.budget_type = budget_type
        self.lower_is_better = lower_is_better

        assert budget_type in self.fidelity_dims, f'The budget type you choose is not available ' \
                                                  f'in this instance. Please choose from ' \
                                                  f'{self.fidelity_dims}.'

        if len(self.fidelity_dims) > 1:
            other_fidelities = [fid for fid in self.fidelity_dims if fid != budget_type]
            self.max_other_fidelities = {}
            for fidelity in other_fidelities:
                self.max_other_fidelities[fidelity] = self.fidelity_space.get_hyperparameter(
                    fidelity).upper


        # TODO on installation of yahpo you need to clone this repo and move it to some_path
        # setting up meta data for surrogate benchmarks
        # from yahpo_gym import local_config
        #
        # local_config.init_config()
        # local_config.set_data_path("some_path")

        self.metric = metric

    @property
    def configspace(self) -> ConfigurationSpace:
        """Return configuration space.

        Returns
        -------
        ConfigurationSpace
            Configuration space.
        """
        return self._configspace

    # @property
    # FIXME: see caro's message:
    #  the idea is somehow to overwrite the optimizer/multifidelity attributes for
    #  budget_variable and min_budget, max_budget with a FidelitiySpace class without interpolation
    #  that is based on the problem instance / config file. Similarily find out how to deal with
    #  the metrics.
    # def fidelity_space(self):
    #     return FidelitySpace(self.fidelity_dims)

    def evaluate(self, trial_info: TrialInfo) -> float:
        """Evaluate problem.

        Parameters
        ----------
        trial_info : TrialInfo
            Dataclass with configuration, seed, budget, instance.

        Returns
        -------
        float
            Cost
        """
        configuration = trial_info.config
        xs = configuration.get_dictionary()

        # fixme: if there are multiple fidelities, we will need to max the respective other
        #  dimension here!

        xs.update({self.budget_type: int(trial_info.budget)})

        if len(self.fidelity_dims) > 1:
            xs.update(self.max_other_fidelities)

        # fixme: figure out why list is returned here!

        if self.lower_is_better:
            return self._problem.objective_function(xs)[0][self.metric]

        else:
            return -self._problem.objective_function(xs)[0][self.metric]




if __name__ == '__main__':









    list_scenarios()

    YahpoProblem.COMBIS



    b = BenchmarkSet(scenario="lcbench")

    b.targets

    b.instances



    # Set an instance
    b.set_instance("3945")

    combi = {}
    fids = {}
    metrics = {}
    for s in list_scenarios():
        b = BenchmarkSet(scenario=s)
        combi[s] = []
        all_fids = set()

        metrics[s] = b.targets

        for i in b.instances:
            combi[s].append(i)
            fid= b.get_fidelity_space()
            all_fids.update(list(k.upper for k in fid._hyperparameters.values()))
            fids[s, i] = list(fid._hyperparameters.keys())
        print(f'{s}: {all_fids}   {len(all_fids)}')

    # Sample a point from the configspace
    search_space = b.get_opt_space()  # NOTICE this has drop_fidelity_params=True argument
    search_space

    xs = search_space.sample_configuration(1)
    xs

    # evaluate
    b.objective_function(xs)

    # setting up meta data
    from yahpo_gym import local_config

    local_config.init_config()
    local_config.set_data_path("~/PycharmProjects/SMACBenchmarking/bench_data/yahpo_data")


    # when using drop fidelity option:
    b = BenchmarkSet("lcbench", instance="3945")
    # Sample a point from the configspace
    xs = b.get_opt_space(drop_fidelity_params=True).sample_configuration(1)
    # Convert to dictionary and add epoch
    xs = xs.get_dictionary()
    xs.update({'epoch': 52})
    xs
    b.objective_function(xs)

