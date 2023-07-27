import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from pandas.util import testing as pdt

from .. import transition
from ...utils import testing as ust


@pytest.fixture
def basic_df():
    return pd.DataFrame(
        {'x': range(5),
         'y': range(5, 10)},
        index=range(100, 105))


@pytest.fixture
def year():
    return 2112


@pytest.fixture
def totals_col():
    return 'total'


@pytest.fixture
def rates_col():
    return 'growth_rate'


@pytest.fixture
def grow_targets(year, totals_col):
    return pd.DataFrame({totals_col: [7]}, index=[year])


@pytest.fixture
def grow_targets_filters(year, totals_col):
    return pd.DataFrame({'x_min': [0, 2, np.nan],
                         'y_max': [7, 9, np.nan],
                         'x': [np.nan, np.nan, 4],
                         totals_col: [1, 4, 10]},
                        index=[year, year, year])


@pytest.fixture(scope='function')
def random_df(request):
    """
    Seed the numpy prng and return a data frame w/ predictable test inputs
    so that the tests will have consistent results across builds.
    """
    old_state = np.random.get_state()

    def fin():
        # tear down: reset the prng after the test to the pre-test state
        np.random.set_state(old_state)

    request.addfinalizer(fin)
    np.random.seed(1)
    return pd.DataFrame(
        {'some_count': np.random.randint(1, 8, 20)},
        index=range(0, 20))


@pytest.fixture
def growth_rates(rates_col, totals_col, grow_targets):
    del grow_targets[totals_col]
    grow_targets[rates_col] = [0.4]
    return grow_targets


@pytest.fixture
def growth_rates_filters(rates_col, totals_col, grow_targets_filters):
    del grow_targets_filters[totals_col]
    grow_targets_filters[rates_col] = [0.5, -0.5, 0]
    return grow_targets_filters


def assert_empty_index(index):
    pdt.assert_index_equal(index, pd.Index([]), exact=False)


def assert_for_add(new, added):
    assert len(new) == 7
    pdt.assert_index_equal(added, pd.Index([105, 106]))


def assert_for_remove(new, added):
    assert len(new) == 3
    assert_empty_index(added)


def test_add_rows(basic_df):
    nrows = 2
    new, added, copied = transition.add_rows(basic_df, nrows)
    assert_for_add(new, added)
    assert len(copied) == nrows
    assert copied.isin(basic_df.index).all()


def test_add_rows_starting_index(basic_df):
    nrows = 2
    starting_index = 1000
    new, added, copied = transition.add_rows(basic_df, nrows, starting_index)
    assert len(new) == len(basic_df) + nrows
    pdt.assert_index_equal(added, pd.Index([1000, 1001]))
    assert len(copied) == nrows
    assert copied.isin(basic_df.index).all()


def test_add_rows_zero(basic_df):
    nrows = 0
    new, added, copied = transition.add_rows(basic_df, nrows)
    pdt.assert_frame_equal(new, basic_df)
    assert_empty_index(added)
    assert_empty_index(copied)


def test_add_rows_with_accounting(random_df):
    control = 10
    new, added, copied = transition.add_rows(
        random_df, control, accounting_column='some_count')
    assert control == new.loc[copied]['some_count'].sum()
    assert copied.isin(random_df.index).all()


def test_remove_rows(basic_df):
    nrows = 2
    new, removed_indexes = transition.remove_rows(basic_df, nrows)
    assert_for_remove(new, transition._empty_index())
    assert len(removed_indexes) == nrows
    assert removed_indexes.isin(basic_df.index).all()


def test_remove_rows_zero(basic_df):
    nrows = 0
    new, removed = transition.remove_rows(basic_df, nrows)
    pdt.assert_frame_equal(new, basic_df)
    assert_empty_index(removed)


def test_remove_rows_all(basic_df):
    nrows = len(basic_df)
    new, removed = transition.remove_rows(basic_df, nrows)
    pdt.assert_frame_equal(new, basic_df.loc[[]], check_index_type=False)
    ust.assert_index_equal(removed, basic_df.index)


def test_remove_rows_with_accounting(random_df):
    control = 10
    new, removed = transition.remove_rows(
        random_df, control, accounting_column='some_count')
    assert control == random_df.loc[removed]['some_count'].sum()
    assert removed.isin(random_df.index).all()


def test_remove_rows_raises(basic_df):
    # should raise ValueError if asked to remove more rows than
    # are in the table
    nrows = 25

    with pytest.raises(ValueError):
        transition.remove_rows(basic_df, nrows)


def test_add_or_remove_rows_add(basic_df):
    nrows = 2
    new, added, copied, removed = \
        transition.add_or_remove_rows(basic_df, nrows)
    assert_for_add(new, added)
    assert len(copied) == abs(nrows)
    assert copied.isin(basic_df.index).all()
    assert_empty_index(removed)


def test_add_or_remove_rows_remove(basic_df):
    nrows = -2
    new, added, copied, removed = \
        transition.add_or_remove_rows(basic_df, nrows)
    assert_for_remove(new, added)
    assert len(removed) == abs(nrows)
    assert removed.isin(basic_df.index).all()
    assert_empty_index(copied)


def test_add_or_remove_rows_zero(basic_df):
    nrows = 0
    new, added, copied, removed = \
        transition.add_or_remove_rows(basic_df, nrows)
    pdt.assert_frame_equal(new, basic_df)
    assert_empty_index(added)
    assert_empty_index(copied)
    assert_empty_index(removed)


def test_grtransition_add(basic_df):
    growth_rate = 0.4
    year = 2112
    grt = transition.GrowthRateTransition(growth_rate)
    new, added, copied, removed = grt.transition(basic_df, year)
    assert_for_add(new, added)
    assert len(copied) == 2
    assert copied.isin(basic_df.index).all()
    assert_empty_index(removed)


def test_grtransition_add_with_accounting(random_df):
    growth_rate = .1
    year = 2012
    orig_total = random_df['some_count'].sum()
    growth = int(round(orig_total * growth_rate))
    target = orig_total + growth
    grt = transition.GrowthRateTransition(growth_rate, 'some_count')
    new, added, copied, removed = grt(random_df, year)
    assert growth == new.loc[copied]['some_count'].sum()
    assert target == new['some_count'].sum()
    assert copied.isin(random_df.index).all()
    assert_empty_index(removed)


def test_grtransition_remove(basic_df):
    growth_rate = -0.4
    year = 2112
    grt = transition.GrowthRateTransition(growth_rate)
    new, added, copied, removed = grt.transition(basic_df, year)
    assert_for_remove(new, added)
    assert_empty_index(copied)
    assert len(removed) == 2
    assert removed.isin(basic_df.index).all()


def test_grtransition_remove_with_accounting(random_df):
    growth_rate = -.1
    year = 2012
    orig_total = random_df['some_count'].sum()
    change = -1 * int(round(orig_total * growth_rate))
    target = orig_total - change
    grt = transition.GrowthRateTransition(growth_rate, 'some_count')
    new, added, copied, removed = grt(random_df, year)
    assert change == random_df.loc[removed]['some_count'].sum()
    assert target == new['some_count'].sum()
    assert removed.isin(random_df.index).all()
    assert_empty_index(added)
    assert_empty_index(copied)


def test_grtransition_remove_all(basic_df):
    growth_rate = -1
    year = 2112
    grt = transition.GrowthRateTransition(growth_rate)
    new, added, copied, removed = grt.transition(basic_df, year)
    pdt.assert_frame_equal(new, basic_df.loc[[]], check_index_type=False)
    assert_empty_index(added)
    assert_empty_index(copied)
    ust.assert_index_equal(removed, basic_df.index)


def test_grtransition_zero(basic_df):
    growth_rate = 0
    year = 2112
    grt = transition.GrowthRateTransition(growth_rate)
    new, added, copied, removed = grt.transition(basic_df, year)
    pdt.assert_frame_equal(new, basic_df)
    assert_empty_index(added)
    assert_empty_index(copied)
    assert_empty_index(removed)


def test_tgrtransition_add(basic_df, growth_rates, year, rates_col):
    tgrt = transition.TabularGrowthRateTransition(growth_rates, rates_col)
    new, added, copied, removed = tgrt.transition(basic_df, year)
    assert len(new) == 7
    bdf_imax = basic_df.index.values.max()
    assert pd.Series([bdf_imax + 1, bdf_imax + 2]).isin(new.index).all()
    assert len(copied) == 2
    assert_empty_index(removed)


def test_tgrtransition_remove(basic_df, growth_rates, year, rates_col):
    growth_rates[rates_col] = -0.4
    tgrt = transition.TabularGrowthRateTransition(growth_rates, rates_col)
    new, added, copied, removed = tgrt.transition(basic_df, year)
    assert len(new) == 3
    assert_empty_index(added)
    assert_empty_index(copied)
    assert len(removed) == 2


def test_tgrtransition_with_accounting(random_df):
    """
    Test segmented growth rate transitions--with an accounting
    column--using 1 test w/ mixed growth rates trends:
    declining, growing and no growth.
    """
    grp1 = random_df.copy()
    grp1['segment'] = 'a'
    grp2 = random_df.copy()
    grp2['segment'] = 'b'
    grp3 = random_df.copy()
    grp3['segment'] = 'c'
    test_df = pd.concat([grp1, grp2, grp3], axis=0, ignore_index=True)
    orig_total = random_df['some_count'].sum()

    year = 2012
    growth_rates = pd.DataFrame(
        {
            'grow_rate': [-0.1, 0.25, 0],
            'segment': ['a', 'b', 'c']
        },
        index=[year, year, year])
    tgrt = transition.TabularGrowthRateTransition(
        growth_rates, 'grow_rate', 'some_count')
    new, added, copied, removed = tgrt.transition(test_df, year)
    added_rows = new.loc[copied]
    removed_rows = test_df.loc[removed]

    # test a declining segment
    a_added_rows = added_rows[added_rows['segment'] == 'a']
    a_removed_rows = removed_rows[removed_rows['segment'] == 'a']
    a_change = int(round(orig_total * -0.1))
    a_target = orig_total + a_change
    assert a_change * -1 == a_removed_rows['some_count'].sum()
    assert a_target == new[new['segment'] == 'a']['some_count'].sum()
    assert_empty_index(a_added_rows.index)

    # test a growing segment
    b_added_rows = added_rows[added_rows['segment'] == 'b']
    b_removed_rows = removed_rows[removed_rows['segment'] == 'b']
    b_change = int(round(orig_total * 0.25))
    b_target = orig_total + b_change
    assert b_change == b_added_rows['some_count'].sum()
    assert b_target == new[new['segment'] == 'b']['some_count'].sum()
    assert_empty_index(b_removed_rows.index)

    # test a no change segment
    c_added_rows = added_rows[added_rows['segment'] == 'c']
    c_removed_rows = removed_rows[removed_rows['segment'] == 'c']
    assert orig_total == new[new['segment'] == 'c']['some_count'].sum()
    assert_empty_index(c_added_rows.index)
    assert_empty_index(c_removed_rows.index)


def test_tgrtransition_remove_all(basic_df, growth_rates, year, rates_col):
    growth_rates[rates_col] = -1
    tgrt = transition.TabularGrowthRateTransition(growth_rates, rates_col)
    new, added, copied, removed = tgrt.transition(basic_df, year)
    pdt.assert_frame_equal(new, basic_df.loc[[]], check_index_type=False)
    assert_empty_index(added)
    assert_empty_index(copied)
    ust.assert_index_equal(removed, basic_df.index)


def test_tgrtransition_zero(basic_df, growth_rates, year, rates_col):
    growth_rates[rates_col] = 0
    tgrt = transition.TabularGrowthRateTransition(growth_rates, rates_col)
    new, added, copied, removed = tgrt.transition(basic_df, year)
    pdt.assert_frame_equal(new, basic_df)
    assert_empty_index(added)
    assert_empty_index(copied)
    assert_empty_index(removed)


def test_tgrtransition_filters(
        basic_df, growth_rates_filters, year, rates_col):
    tgrt = transition.TabularGrowthRateTransition(
        growth_rates_filters, rates_col)
    new, added, copied, removed = tgrt.transition(basic_df, year)
    assert len(new) == 5
    assert basic_df.index.values.max() + 1 in new.index
    assert len(copied) == 1
    assert len(removed) == 1


def test_tabular_transition_add(basic_df, grow_targets, totals_col, year):
    tran = transition.TabularTotalsTransition(grow_targets, totals_col)
    new, added, copied, removed = tran.transition(basic_df, year)
    assert_for_add(new, added)
    bdf_imax = basic_df.index.values.max()
    assert pd.Series([bdf_imax + 1, bdf_imax + 2]).isin(new.index).all()
    assert len(copied) == 2
    assert_empty_index(removed)


def test_tabular_transition_remove(basic_df, grow_targets, totals_col, year):
    grow_targets[totals_col] = [3]
    tran = transition.TabularTotalsTransition(grow_targets, totals_col)
    new, added, copied, removed = tran.transition(basic_df, year)
    assert_for_remove(new, added)
    assert_empty_index(copied)
    assert len(removed) == 2


def test_tabular_transition_remove_all(
        basic_df, grow_targets, totals_col, year):
    grow_targets[totals_col] = [0]
    tran = transition.TabularTotalsTransition(grow_targets, totals_col)
    new, added, copied, removed = tran.transition(basic_df, year)
    pdt.assert_frame_equal(new, basic_df.loc[[]], check_index_type=False)
    assert_empty_index(added)
    assert_empty_index(copied)
    ust.assert_index_equal(removed, basic_df.index)


def test_tabular_transition_raises_on_bad_year(
        basic_df, grow_targets, totals_col, year):
    tran = transition.TabularTotalsTransition(grow_targets, totals_col)

    with pytest.raises(ValueError):
        tran.transition(basic_df, year + 100)


def test_tabular_transition_add_filters(
        basic_df, grow_targets_filters, totals_col, year):
    tran = transition.TabularTotalsTransition(grow_targets_filters, totals_col)
    new, added, copied, removed = tran.transition(basic_df, year)

    assert len(new) == grow_targets_filters[totals_col].sum()
    assert basic_df.index.values.max() + 1 in new.index
    assert len(copied) == 11
    assert len(removed) == 1


def test_update_linked_table(basic_df):
    col_name = 'x'
    added = pd.Index([5, 6, 7])
    copied = pd.Index([1, 3, 1])
    removed = pd.Index([0])

    updated = transition._update_linked_table(
        basic_df, col_name, added, copied, removed)

    assert len(updated) == len(basic_df) + len(added) - len(removed)
    npt.assert_array_equal(updated[col_name].values, [1, 2, 3, 4, 5, 7, 6])
    pdt.assert_series_equal(
        updated['y'],
        pd.Series([6, 7, 8, 9, 6, 6, 8], index=updated.index, name='y'),
        check_dtype=False)


def test_updated_linked_table_remove_only(basic_df):
    col_name = 'x'
    added = pd.Index([])
    copied = pd.Index([])
    removed = pd.Index([1, 3])

    updated = transition._update_linked_table(
        basic_df, col_name, added, copied, removed)
    assert len(updated) == len(basic_df) + len(added) - len(removed)


def test_transition_model(basic_df, grow_targets_filters, totals_col, year):
    grow_targets_filters[totals_col] = [3, 1, 1]
    tran = transition.TabularTotalsTransition(grow_targets_filters, totals_col)
    model = transition.TransitionModel(tran)

    linked_table = pd.DataFrame(
        {'z': ['a', 'b', 'c', 'd', 'e'],
         'thing_id': basic_df.index})

    new, added, new_linked = model.transition(
        basic_df, year, linked_tables={'linked': (linked_table, 'thing_id')})

    assert len(new) == grow_targets_filters[totals_col].sum()
    assert new.index.values.max() == basic_df.index.values.max() + 1
    assert len(new_linked['linked']) == grow_targets_filters[totals_col].sum()
    assert new.index.values.max() in new_linked['linked'].thing_id.values
    assert new_linked['linked'].index.values.max() == 5
    assert added.isin(new.index).all()
    assert not added.isin(basic_df.index).any()
    npt.assert_array_equal(added.values, [basic_df.index.values.max() + 1])


def test_tabular_transition_add_and_remove():
    data = pd.DataFrame(
        {'a': ['x', 'x', 'y', 'y', 'y', 'y', 'y', 'y', 'z', 'z']})

    totals = pd.DataFrame(
        {'a': ['x', 'y', 'z'],
         'total': [3, 1, 10]},
        index=[2112, 2112, 2112])

    tran = transition.TabularTotalsTransition(totals, 'total')
    model = transition.TransitionModel(tran)

    new, added, _ = model.transition(data, 2112)

    assert len(new) == totals.total.sum()
    assert added.is_unique is True
    assert new.index.is_unique is True


################################
# TESTS FOR SAMPLING THRESHOLDS
################################


@pytest.fixture
def st_df():
    """
    Data for testing sampling thresholds.

    """
    return pd.DataFrame(
        [
            # case 1 - no agents -- query by city
            ('Maricopa', 'Chandler', 1,  20, 0),
            
            # case 2 -- min threshold met -- query by taz
            ('Maricopa', 'Chandler', 2, 200, 101),

            # case 3 -- min theshold not met -- but has a decline -- query by taz
            ('Maricopa', 'Chandler', 3, 30, 40),

            # case 4 -- min threshold no met -- query by county
            ('Maricopa', 'Tempe', 4, 90, 50),

            # case 5 -- min threshold not net -- query the entire dataset
            ('Pinal', 'Apache Junction', 5, 120, 95),

            # case 6 -- agents w/out segments in controls table
            ('NOT CONTROLLED', '', -1, 0, 79)
        ],
        columns=['county', 'city', 'taz', 'control', 'agent_cnt']
    )


@pytest.fixture
def st_controls(st_df, year):
    """
    Controls for testing sampling thresholds.

    """
    controls = st_df.drop(['agent_cnt'], axis=1).query("county != 'NOT CONTROLLED'")
    controls['year'] = year
    controls.set_index('year', inplace=True)
    return controls


@pytest.fixture
def st_controls_accounting(st_controls):
    st_controls['control'] *= 2
    return st_controls


@pytest.fixture
def st_agents(st_df):
    """
    Agents for testing sampling thresholds

    """
    agents = st_df.loc[st_df.index.repeat(st_df['agent_cnt'])].drop(['agent_cnt', 'control'], axis=1)
    agents.reset_index(inplace=True, drop=True)
    agents.index.name = 'agent_id'
    agents_bak = agents.copy()

    # for testing accounting column
    agents['amount'] = 2

    return agents


@pytest.fixture
def st_sampling_hierarchy():
    """
    Sampling hiearchy to test with. Should be ordered
    from left to right, from least to most detiled.

    """
    return ['county', 'city', 'taz']


def check_st_controls(agents, controls,sampling_hierarchy):
    """
    Check the agents counts vs. the controls.

    """
    m = pd.concat({
        'a': agents.groupby(sampling_hierarchy).size(),
        'c': controls.groupby(sampling_hierarchy)['control'].sum(),
    }, axis=1).fillna(-9999)
    assert (m['a'] == m['c']).all()


def test_st_count(year, st_controls, st_agents, st_sampling_hierarchy):
    """
    Test sampling thresholds when threshold is expressed as a count.

    """
    min_threshold = 100
    tm = transition.TabularTotalsTransition(
        st_controls, 
        'control', 
        sampling_threshold=min_threshold, 
        sampling_hierarchy=st_sampling_hierarchy
    )
    updated, added, copied, removed = tm.transition(st_agents, year)
    check_st_controls(updated, st_controls, st_sampling_hierarchy)


def test_st_count_runout_segments(year, st_controls, st_agents, st_sampling_hierarchy):
    """
    Test sampling thresholds when threshold is expressed as a count, 
    when running out segments, occurs when threshold is larger than 
    the agents in the dataset.

    """
    min_threshold = 10000
    tm = transition.TabularTotalsTransition(
        st_controls, 
        'control', 
        sampling_threshold=min_threshold, 
        sampling_hierarchy=st_sampling_hierarchy
    )
    updated, added, copied, removed = tm.transition(st_agents, year)
    check_st_controls(updated, st_controls, st_sampling_hierarchy)


def test_st_keep_outside(year, st_controls, st_agents, st_sampling_hierarchy):
    """
    Test w/ the `keep_outside` option set to True. This has the affect 
    of retaining agents not covered by the control segmentation

    """
    min_threshold = 100
    tm = transition.TabularTotalsTransition(
        st_controls, 
        'control', 
        sampling_threshold=min_threshold, 
        sampling_hierarchy=st_sampling_hierarchy,
        keep_outside=True
    )
    updated, added, copied, removed = tm.transition(st_agents, year)
    assert len(updated.query("county == 'NOT CONTROLLED'")) == len(st_agents.query("county == 'NOT CONTROLLED'"))
    check_st_controls(
        updated.query("county != 'NOT CONTROLLED'"), st_controls, st_sampling_hierarchy)


def test_st_ratio(year, st_controls, st_agents, st_sampling_hierarchy):
    """
    Test sampling thresholds when threshold is expressed as the maximum
    ratio between the rows being added and the existing row counts. 

    """
    threshold = 0.3
    tm = transition.TabularTotalsTransition(
        st_controls, 
        'control', 
        sampling_threshold=threshold, 
        sampling_hierarchy=st_sampling_hierarchy
    )
    updated, added, copied, removed = tm.transition(st_agents, year)
    check_st_controls(updated, st_controls, st_sampling_hierarchy)


def check_st_controls_accounting(agents, controls, sampling_hierarchy):
    """
    Check the agents counts vs. the controls.

    """
    m = pd.concat({
        'a': agents.groupby(sampling_hierarchy)['amount'].sum(),
        'c': controls.groupby(sampling_hierarchy)['control'].sum(),
    }, axis=1).fillna(-9999)
    assert (m['a'] == m['c']).all()


def test_st_count_w_accounting(year, st_controls_accounting, st_agents, st_sampling_hierarchy):
    """
    Test sampling thresholds when threshold is expressed as a count,
    when using accounting column.

    """
    min_threshold = 100
    tm = transition.TabularTotalsTransition(
        st_controls_accounting, 
        'control',
        accounting_column='amount',
        sampling_threshold=min_threshold, 
        sampling_hierarchy=st_sampling_hierarchy
    )
    updated, added, copied, removed = tm.transition(st_agents, year)
    check_st_controls_accounting(updated, st_controls_accounting, st_sampling_hierarchy)


def test_st_ratio_w_accounting(year, st_controls_accounting, st_agents, st_sampling_hierarchy):
    """
    Test sampling thresholds when threshold is expressed as the maximum
    ratio between the rows being added and the existing row counts, 
    when using accounting column.

    """
    threshold = 0.3
    tm = transition.TabularTotalsTransition(
        st_controls_accounting, 
        'control',
        accounting_column='amount',
        sampling_threshold=threshold, 
        sampling_hierarchy=st_sampling_hierarchy
    )
    updated, added, copied, removed = tm.transition(st_agents, year)
    check_st_controls_accounting(updated, st_controls_accounting, st_sampling_hierarchy)
