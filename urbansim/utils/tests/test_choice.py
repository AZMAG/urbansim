import numpy as np
import pandas as pd
import pytest

from .. choice import *


##################
# GENERAL TESTS
##################


def assert_arr(arr1, arr2):
    """
    Asserts that all elements in 2 arrays are equal.

    """
    assert (arr1 == arr2).all()


def test_get_probs():
    # typical case, note the nan will be a 0
    s = pd.Series([8, 6, 4, 2, np.nan])
    p = get_probs(s)
    assert_arr(p.values, [.4, .3, .2, .1, 0])

    # now test a case with a zero weight sum
    s = pd.Series(np.zeros(4))
    p = get_probs(s)
    assert_arr(p.values, 0.25)


def test_get_segmented_probs():
    df = pd.DataFrame({
        'grp': ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
        'w': [1, 3, 6, 1, np.nan, 0, 0, np.nan, np.nan]
    })
    p = get_segmented_probs(df, 'w', 'grp')
    assert_arr(p.values, [.1, .3, .6, 1, 0, 0.5, 0.5, 0.5, 0.5])


def test_randomize_probs():
    # the random seed to apply
    seed = 123

    # get the function results
    w = pd.Series([2, 4, 6])
    p = get_probs(w)
    r_p = seeded_call(seed, randomize_probs, p)

    # make sure we can reproduce these
    def get_r(count):
        return np.random.rand(count)

    r = seeded_call(seed, get_r, len(p))
    res = np.power(r, 1.0 / p)
    assert (r_p.values == res).all()


def test_segmented_sample_no_replace():
    seed = 123

    df = pd.DataFrame({
        'grp': ['a', 'a', 'a', 'b', 'b', 'b'],
        'w': [20, 30, 10, 50, 5, 15]
    })

    amounts = pd.Series([1, 2], index=pd.Index(['b', 'a']))

    # 1st test random (i.e. no weights), expected randoms are
    # [ 0.69646919,  0.28613933,  0.22685145,  0.55131477,  0.71946897, 0.42310646]
    ran_sample = seeded_call(
        seed,
        segmented_sample_no_replace,
        amounts, df, 'grp'
    )
    assert (ran_sample == [4, 0, 1]).all()

    # test with weights
    # probs are
    # [0.333333,0.500000,0.166667, 0.714286, 0.071429, 0.214286]
    # randmized probs with this seed should be
    # [0.337836,0.081876, 0.000136, 0.434470, 0.009958, 0.018062]
    # so the sort order should be
    # [3, 0, 1, 5, 4, 2]
    w_sample = seeded_call(
        seed,
        segmented_sample_no_replace,
        amounts, df, ['grp'], 'w'
    )
    assert (w_sample == [3, 0, 1]).all()


def test_get_capacities():

    df = pd.DataFrame({
        'cap': [0, 1, 2]
    })

    test1 = get_capacities(df)
    test2 = get_capacities(df, df.cap)
    test3 = get_capacities(df, 'cap')

    assert_arr(test1.values, 1)
    assert_arr(test2.values, [0, 1, 2])
    assert_arr(test3.values, [0, 1, 2])


####################
# AGGREGATE CHOICES
####################


@pytest.fixture()
def agg_choosers():
    return pd.DataFrame(
        {
            'col': ['a', 'b', 'c'],
            'grp': [1, 1, 2]
        },
        index=pd.Index([16, 12, 10])
    )


@pytest.fixture()
def agg_alts():
    return pd.DataFrame(
        {
            'w': [1, 5, 10, 2],
            'c': [3, 2, 1, 0],
            'grp': [1, 2, 1, 2]
        },
        pd.Index(['b1', 'c1', 'a1', 'z1'])
    )


def test_simple_aggregate_choice(agg_choosers, agg_alts):
    c, p = seeded_call(
        123, simple_aggregate_choice, agg_choosers, agg_alts['w'])

    c = c['alternative_id']
    p = p['prob']
    assert_arr(c.index.values, [16, 12, 10])
    assert_arr(p.index.values, ['b1', 'c1', 'a1', 'z1'])
    assert_arr(c.values, ['a1', 'c1', 'c1'])
    assert_arr(np.round(p.values, 2), [.06, .28, .56, .11])


def test_constrained_aggregate_choice(agg_choosers, agg_alts):

    # test 1 - with capacities and unit normalization
    c, p = seeded_call(
        123, constrained_aggregate_choice, agg_choosers, agg_alts['w'], agg_alts['c'])
    c = c['alternative_id']
    p = p['prob']

    assert_arr(c.index.values, [16, 12, 10])
    assert_arr(p.index.values, ['b1', 'c1', 'a1', 'z1'])
    assert_arr(c.values, ['a1', 'c1', 'c1'])
    assert_arr(p.values, [.0625, .3125, .6250, 0])

    # test 2 - with capacities but NO normalization
    c, p = seeded_call(
        123, constrained_aggregate_choice,
        agg_choosers, agg_alts['w'], agg_alts['c'], False)
    c = c['alternative_id']
    p = p['prob']

    assert_arr(c.index.values, [16, 12, 10])
    assert_arr(p.index.values, ['b1', 'c1', 'a1', 'z1'])
    assert_arr(c.values, ['a1', 'c1', 'c1'])
    assert_arr(np.round(p.values, 2), [.13, .43, .43, 0])

    # test 3 - no capacities provided
    c, p = seeded_call(
        123, constrained_aggregate_choice, agg_choosers, agg_alts['w'])
    c = c['alternative_id']
    p = p['prob']

    assert_arr(c.index.values, [16, 12, 10])
    assert_arr(p.index.values, ['b1', 'c1', 'a1', 'z1'])
    assert_arr(c.values, ['a1', 'c1', 'z1'])
    assert_arr(np.round(p.values, 2), [.06, .28, .56, .11])

    # test 4 - not enough alternatives
    agg_alts['cap2'] = np.array([2, 0, 0, 0])
    c, p = seeded_call(
        123, constrained_aggregate_choice, agg_choosers, agg_alts['w'], agg_alts['cap2'])
    c = c['alternative_id'].fillna('-1')
    p = p['prob']

    assert_arr(c.index.values, [16, 12, 10])
    assert_arr(p.index.values, ['b1', 'c1', 'a1', 'z1'])
    assert_arr(c.values, ['b1', 'b1', '-1'])
    assert_arr(p.values, [1, 0, 0, 0])


def test_aggregate_choice(agg_choosers, agg_alts):

    def w_callback(alternatives):
        return alternatives['w']

    def check(c, p, expected_c, expected_p):
        assert_arr(c.index.values, [16, 12, 10])
        assert_arr(p.index.values, ['b1', 'c1', 'a1', 'z1'])
        assert_arr(c['alternative_id'].values, expected_c)
        assert_arr(np.round(p['prob'].values, 2), expected_p)

    # NON-SEGMENTED TESTS (use unconstrained)
    expected_c = ['a1', 'c1', 'c1']
    expected_p = [.06, .28, .56, .11]

    # test 1, weights as a series
    c, p = seeded_call(
        123, aggregate_choice, agg_choosers, agg_alts, agg_alts['w'], False)
    check(c, p, expected_c, expected_p)

    # test 2, weights as a column name
    c, p = seeded_call(
        123, aggregate_choice, agg_choosers, agg_alts, 'w', False)
    check(c, p, expected_c, expected_p)

    # test 3, weights as a callback function
    c, p = seeded_call(
        123, aggregate_choice, agg_choosers, agg_alts, w_callback, False)
    check(c, p, expected_c, expected_p)

    # SEGMENTED TESTS (use constrained)
    expected_c = ['a1', 'b1', 'c1']
    expected_p = [0.09, 1, 0.91, 0]

    # test 1, weights as a series
    c, p = seeded_call(
        123, aggregate_choice,
        agg_choosers, agg_alts,
        agg_alts['w'], True, segment_col='grp', capacities=agg_alts['c'])
    check(c, p, expected_c, expected_p)

    # test 2, weights as a column name
    c, p = seeded_call(
        123, aggregate_choice,
        agg_choosers, agg_alts,
        'w', True, segment_col='grp', capacities=agg_alts['c'])
    check(c, p, expected_c, expected_p)

    # test 3, weights as a callback function
    c, p = seeded_call(
        123, aggregate_choice,
        agg_choosers, agg_alts,
        w_callback, True, segment_col='grp', capacities=agg_alts['c'])
    check(c, p, expected_c, expected_p)


####################
# INDIVIUDAL CHOICE
####################


@pytest.fixture()
def choosers():
    return pd.DataFrame(
        {
            'agent_col1': [10, 20, 30],
            'choice': [1, 2, 0],
            'grp': [1, 1, 2]
        },
        index=pd.Index(list('cba'))
    )


@pytest.fixture()
def alternatives():
    return pd.DataFrame(
        {
            'alt_col1': [100, 200, 300, 400, 500, 600],
            'cap': [2, 2, 0, 1, 1, 1],
            'cap2': [1, 0, 0, 0, 0, 0],
            'grp': [1, 2, 1, 2, 1, 2]
        },
        index=pd.Index(np.arange(6, 0, -1,))
    )


def test_sample_alts():
    alts = pd.Series(np.arange(5))
    choosers = pd.Series(np.arange(5))
    choices = pd.Series([3, 0, 1, 1, 2])

    # test w/out sampling
    expected_sample = np.array([
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
    ])
    expected_chosen = np.array([
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
    ])
    sample, num_cols, chosen = sample_alts_2d(alts, choosers, observed=choices)
    assert_arr(sample, expected_sample)
    assert num_cols == 5
    assert_arr(chosen, expected_chosen)

    # test w/ more samples than alternatives
    sample, num_cols, chosen = sample_alts_2d(alts, choosers, sample_size=10, observed=choices)
    assert_arr(sample, expected_sample)
    assert num_cols == 5
    assert_arr(chosen, expected_chosen)

    # test w/ sampling and w/ observed
    expected_sample = np.array([
        [0, 2, 3, 4],
        [0, 1, 2, 3],
        [0, 1, 3, 4],
        [0, 1, 2, 4],
        [0, 1, 2, 4]
    ])
    expected_chosen = np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])
    sample, num_cols, chosen = seeded_call(
        123,
        sample_alts_2d,
        alts, choosers, sample_size=3, observed=choices
    )
    assert_arr(sample, expected_sample)
    assert num_cols == 4
    assert_arr(chosen, expected_chosen)

    # sampling without observed
    expected_sample = np.array([
        [2, 3, 4],
        [1, 2, 3],
        [1, 3, 4],
        [0, 1, 4],
        [0, 1, 3]
    ])
    sample, num_cols = seeded_call(
        123,
        sample_alts_2d,
        alts, choosers, 3
    )
    assert_arr(sample, expected_sample)
    assert num_cols == 3

    # sample with weights
    weights = pd.Series([0, 1, 100, 100, 50])
    expected_sample = np.array([
        [2, 3],
        [2, 3],
        [2, 3],
        [3, 4],
        [2, 3]
    ])
    sample, num_cols = seeded_call(
        123,
        sample_alts_2d,
        alts, choosers, 2, sampling_weights=weights
    )
    assert_arr(sample, expected_sample)
    assert num_cols == 2


def test_sample_no_compete(alternatives, choosers):
    capacities = alternatives['cap']

    # test 1 - with capacity column
    sample, size = seeded_call(
        123, sample_2d_no_compete,
        alternatives, choosers, sample_size=5, capacities=capacities)

    assert size == 2
    expected_sample = np.array([
        [6, 5],
        [3, 6],
        [5, 2]
    ])
    assert_arr(sample, expected_sample)

    # test 2 - no capacity column
    sample, size = seeded_call(
        123, sample_2d_no_compete,
        alternatives, choosers, sample_size=5)

    assert size == 2
    expected_sample = np.array([
        [5, 3],
        [2, 6],
        [4, 1]
    ])
    assert_arr(sample, expected_sample)

    # test 3  - not enough capacity
    sample, size = sample_2d_no_compete(
        alternatives.head(1), choosers,
        sample_size=5, capacities=capacities.head(1))
    assert size == -1

    expected_sample = np.array([
        [6],
        [6]
    ])
    assert_arr(sample, expected_sample)


def assert_interact(data, alternatives, choosers, size):
    for cc in choosers.columns:
        if cc in alternatives.columns:
            assert '{}_chooser'.format(cc) in data.columns
        else:
            assert cc in data.columns
    for ac in alternatives.columns:
        if ac in choosers.columns:
            assert '{}_alt'.format(ac) in data.columns
        else:
            assert ac in data.columns
    assert len(data) == len(choosers) * size

    choosers_idx = data.index.get_level_values(0).values

    assert_arr(choosers_idx, np.repeat(choosers.index.values, size))


def test_interaction_data_estimation(choosers, alternatives):
    data, num_samples, chosen = get_interaction_data_for_estimation(
        choosers, alternatives, choosers['choice'], 3)
    assert_interact(data, alternatives, choosers, num_samples)


def test_interaction_data_predict(choosers, alternatives):
    data, num_samples, num_choosers = get_interaction_data_for_predict(
        choosers, alternatives, 3)
    assert_interact(data, alternatives, choosers, num_samples)

    # non competing
    data, num_samples, num_choosers = get_interaction_data_for_predict(
        choosers, alternatives, 3, False, alternatives['cap'])
    assert_interact(data, alternatives, choosers, num_samples)

    # non competing -- not enough capacity
    data, num_samples, num_choosers = get_interaction_data_for_predict(
        choosers, alternatives, 3, False, alternatives['cap2'])
    assert num_samples == 1
    assert len(data) == 1


def test_interaction_bad(choosers, alternatives):

    bad_alts = pd.DataFrame(
        index=pd.Index(np.ones(len(alternatives))))
    bad_choosers = pd.DataFrame(
        index=pd.Index(np.ones(len(choosers))))

    with pytest.raises(ValueError):
        a, b, c = get_interaction_data_for_estimation(
            choosers, bad_alts, choosers['choice'], 3)

    with pytest.raises(ValueError):
        a, b, c = get_interaction_data_for_estimation(
            bad_choosers, alternatives, choosers['choice'], 3)

    with pytest.raises(ValueError):
        a, b, c = get_interaction_data_for_predict(
            choosers, bad_alts, 3)

    with pytest.raises(ValueError):
        a, b, c = get_interaction_data_for_predict(
            bad_choosers, alternatives, 3)


def test_default_sampler(choosers, alternatives):
    num_samples = 2

    d1 = DefaultSampler(sample_size=2)
    data, sample_size, num_choosers = seeded_call(
        123, d1, choosers, alternatives)

    assert_interact(data, alternatives, choosers, num_samples)
    sample_ids = data.index.get_level_values(1).values
    assert_arr(sample_ids, [1, 4, 2, 4, 3, 5])

    d2 = DefaultSampler(sample_size=2, sampling_weights_col='cap')
    data, sample_size, num_choosers = seeded_call(
        123, d2, choosers, alternatives)

    assert_interact(data, alternatives, choosers, num_samples)
    sample_ids = data.index.get_level_values(1).values
    assert_arr(sample_ids, [3, 5, 5, 6, 2, 5])


def test_get_sample_for_segment(choosers, alternatives):

    segment = 1
    segment_col = 'grp'

    # test 1 - no sample size:
    # this will return all alts in the segment
    data, sample_size, num_choosers = get_sample_for_segment(
        choosers, alternatives, segment, segment_col)
    sample_ids = data.index.get_level_values(1).values
    assert num_choosers == 2
    assert sample_size == 3
    assert_arr(sample_ids, [6, 4, 2, 6, 4, 2])

    # test 2 - sample all within the same segment
    data, sample_size, num_choosers = seeded_call(
        123,
        get_sample_for_segment,
        choosers,
        alternatives,
        segment, segment_col, sample_size=2
    )
    assert num_choosers == 2
    assert sample_size == 2
    sample_ids = data.index.get_level_values(1).values
    assert_arr(sample_ids, [2, 4, 2, 6])

    # test 3 - vary the amount within the segment
    data, sample_size, num_choosers = seeded_call(
        123,
        get_sample_for_segment,
        choosers,
        alternatives,
        segment, segment_col, sample_size=2, sampling_within_percent=0.5
    )
    assert num_choosers == 2
    assert sample_size == 2
    sample_ids = data.index.get_level_values(1).values
    assert_arr(sample_ids, [2, 1, 4, 1])


class TestProb():

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, interaction_data):
        # simple probabilities function that just uses the alt column as a weight
        return self.factor * interaction_data['alt_col1']


def test_individual_choice(choosers, alternatives):
    seed = 123
    sample_size = 4
    sampler = DefaultSampler(sample_size)
    prob_call = TestProb(1.0)

    # 1st test w/out verbosity
    expected_choices = np.array([4, 4, 3])
    expected_probs = np.array([.19, .30, .31])

    choices, _ = seeded_call(
        seed,
        individual_choice,
        choosers, alternatives, sampler, prob_call
    )
    assert_arr(choosers.index.values, choices.index.values)
    assert_arr(expected_choices, choices['alternative_id'])
    assert_arr(expected_probs, choices['prob'].round(2))

    # test w/ verbosity
    choices, samples = seeded_call(
        seed,
        individual_choice,
        choosers, alternatives, sampler, prob_call, verbose=True
    )
    assert_arr(samples.index.values, np.repeat(choosers.index.values, sample_size))
    assert samples['prob'].sum() == len(choosers)


def test_segmented_individual_choice(choosers, alternatives):
    prob_call = TestProb(1.0)

    choices, samples = seeded_call(
        123,
        segmented_individual_choice,
        choosers,
        alternatives,
        prob_call,
        segment_col='grp',
        sample_size=2,
        sampling_segments={
            2: 0.5
        },
        verbose=True
    )
    assert_arr(choices.index.values, choosers.index.values)
    assert_arr(choices['alternative_id'].values, [4, 2, 4])
    assert_arr(
        samples['alternative_id'].values, [2, 4, 2, 6, 1, 4])


def test_constrained_individual_choice(choosers, alternatives):
    seed = 123
    sample_size = 4
    prob_call = TestProb(1.0)
    choice_call = IndividualChoiceCallback(prob_call, sample_size=sample_size)

    # 1st test w/out verbosity
    expected_choices = np.array([2, 3, 1])
    expected_capacities = np.array([2, 2, 0, 0, 0, 0])

    choices, capacity, _ = seeded_call(
        seed,
        constrained_choice,
        choosers,
        alternatives,
        choice_call,
        capacity='cap'
    )
    assert_arr(choosers.index.values, choices.index.values)
    assert_arr(expected_choices, choices)
    assert_arr(capacity.index.values, alternatives.index.values)
    assert_arr(capacity, expected_capacities)

    # test w/ verbosity
    choices, capacity, samples = seeded_call(
        seed,
        constrained_choice,
        choosers,
        alternatives,
        choice_call,
        capacity='cap',
        verbose=True
    )
    assert 'alternative_id' in samples.columns
    assert 'prob' in samples.columns
    assert 'iter' in samples.columns

    # test with not enough capacity
    choices, capacity, _ = choices, capacity, _ = seeded_call(
        seed,
        constrained_choice,
        choosers,
        alternatives,
        choice_call,
        capacity='cap2'
    )
    assert_arr(choices.fillna(-1), [-1, 6, -1])
    assert_arr(capacity.values, 0)
