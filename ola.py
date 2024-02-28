from HLL import HyperLogLog
from typing import List, Any

import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict


class OLA:
    def __init__(self, widget: go.FigureWidget):
        """
            Base OLA class.

            *****************************************
            * You do not have to modify this class. *
            *****************************************

            @param widget: The dynamically updating plotly plot.
        """
        self.widget = widget

    def process_slice(df_slice: pd.DataFrame) -> None:
        """
            Process a dataframe slice. To be implemented in inherited classes.
        """
        pass

    def update_widget(self, groups_list: List[Any], values_list: List[Any]) -> None:
        """
            Update the plotly widget with newest groupings and values.

            @param groups_list: List of groups.
            @param values_list: List of grouped values (e.g., grouped means/sums).
        """
        self.widget.data[0]['x'] = groups_list
        self.widget.data[0]['y'] = values_list


class AvgOla(OLA):
    def __init__(self, widget: go.FigureWidget, mean_col: str):
        """
            Class for performing OLA by incrementally computing the estimated mean of *mean_col*.
            This class is implemented for you as an example.

            @param mean_col: column to compute filtered mean for.
        """
        super().__init__(widget)
        self.mean_col = mean_col

        # Bookkeeping variables
        self.sum = 0
        self.count = 0

    def process_slice(self, df_slice: pd.DataFrame) -> None:
        """
            Update the running mean with a data frame slice.
        """
        self.sum += df_slice.sum()[self.mean_col]
        self.count += df_slice.count()[self.mean_col]

        # Update the plot. The mean should be put into a singleton list due to Plotly semantics.
        # Note: there is no x axis label since there is only one bar.
        self.update_widget([""], [self.sum / self.count])


class FilterAvgOla(OLA):
    def __init__(self, plot_widget: go.FigureWidget, filter_column: str, filter_value: Any, target_column: str):
        """
        Class for performing OLA by incrementally computing the estimated filtered mean of *target_column*
        where *filter_column* is equal to *filter_value*.

        @param plot_widget: The dynamically updating plotly plot.
        @param filter_column: Column to filter on.
        @param filter_value: Value to filter for, i.e., df[df[filter_column] == filter_value].
        @param target_column: Column to compute filtered mean for.
        """
        super().__init__(plot_widget)
        self.filter_column = filter_column
        self.filter_value = filter_value
        self.target_column = target_column

        # Bookkeeping variables
        self.filtered_sum = 0
        self.filtered_count = 0

    def process_slice(self, df_slice: pd.DataFrame) -> None:
        """
        Update the running filtered mean with a dataframe slice.
        """
        filt_df = df_slice[df_slice[self.filter_column] == self.filter_value]
        trgt_col_values = filt_df[self.target_column]

        self.filtered_sum += trgt_col_values.sum()
        self.filtered_count += trgt_col_values.count()

        # Update the plot. The filtered mean should be put into a singleton list due to Plotly semantics.
        self.update_widget([""], [self.filtered_sum / self.filtered_count])


class GroupByAvgOla(OLA):
    def __init__(self, plot_widget: go.FigureWidget, group_column: str, target_column: str):
        """
        Class for performing OLA by incrementally computing the estimated grouped means of *target_column*
        with *group_column* as groups.

        @param plot_widget: The dynamically updating plotly plot.
        @param group_column: Grouping column, i.e., df.groupby(group_column).
        @param target_column: Column to compute grouped means for.
        """
        super().__init__(plot_widget)
        self.group_column = group_column
        self.target_column = target_column

        # Bookkeeping variables
        self.grouped_sum = defaultdict(float)
        self.grouped_count = defaultdict(int)

    def process_slice(self, df_slice: pd.DataFrame) -> None:
        """
        Update the running grouped means with a dataframe slice.
        """
        grouped_df = df_slice.groupby(self.group_column)
        for group, group_df in grouped_df:
            trgt_col_val = group_df[self.target_column]

            self.grouped_sum[group] += trgt_col_val.sum()
            self.grouped_count[group] += trgt_col_val.count()

        # Update the plot
        grps = list(self.grouped_sum.keys())
        grp_means = [self.grouped_sum[group] / self.grouped_count[group] for group in grps]
        self.update_widget(grps, grp_means)


class GroupBySumOla(OLA):
    def __init__(self, plot_widget: go.FigureWidget, original_rows: int, group_column: str, sum_column: str):
        """
        Class for performing OLA by incrementally computing the estimated grouped sums of *sum_column*
        with *group_column* as groups.

        @param plot_widget: The dynamically updating plotly plot.
        @param original_rows: Number of rows in the original dataframe before sampling and slicing.
        @param group_column: Grouping column, i.e., df.groupby(group_column).
        @param sum_column: Column to compute grouped sums for.
        """
        super().__init__(plot_widget)
        self.original_rows = original_rows
        self.group_column = group_column
        self.sum_column = sum_column

        # Bookkeeping variables
        self.grouped_sums = defaultdict(float)
        self.total_rows_processed = 0

    def process_slice(self, df_slice: pd.DataFrame) -> None:
        """
        Update the running grouped sums with a dataframe slice.
        """
        groups_in_slice = df_slice.groupby(self.group_column)[self.sum_column].sum()
        self.total_rows_processed += len(df_slice)
        scaling_factor = self.original_rows / self.total_rows_processed

        for group, sum_value in groups_in_slice.items():
            self.grouped_sums[group] += sum_value

        est_sums = [value * scaling_factor for value in self.grouped_sums.values()]

        # Update the plot
        updated_groups = list(self.grouped_sums.keys())
        self.update_widget(updated_groups, est_sums)


class GroupByCountOla(OLA):
    def __init__(self, plot_widget: go.FigureWidget, original_rows: int, group_column: str, count_column: str):
        """
        Class for performing OLA by incrementally computing the estimated grouped counts in *count_column*
        with *group_column* as groups.

        @param plot_widget: The dynamically updating plotly plot.
        @param original_rows: Number of rows in the original dataframe before sampling and slicing.
        @param group_column: Grouping column, i.e., df.groupby(group_column).
        @param count_column: Counting column.
        """
        super().__init__(plot_widget)
        self.original_rows = original_rows
        self.group_column = group_column
        self.count_column = count_column
        self.grouped_counts = defaultdict(int)
        self.rows_processed = 0

    def process_slice(self, df_slice: pd.DataFrame) -> None:
        """
        Update the running grouped counts with a dataframe slice.
        """
        curt_grps = df_slice.groupby(self.group_column)[self.count_column].count().index
        curt_cts = df_slice.groupby(self.group_column)[self.count_column].count()

        self.rows_processed += len(df_slice)
        factor = self.original_rows / self.rows_processed

        for grp, ct in zip(curt_grps, curt_cts):
            self.grouped_counts[grp] += ct

        est_cts = [count * factor for count in self.grouped_counts.values()]
        self.update_widget(list(self.grouped_counts.keys()), est_cts)


class FilterDistinctOla(OLA):
    def __init__(self, widget: go.FigureWidget, filter_col: str, filter_value: Any, distinct_col: str):
        """
        Class for performing OLA by incrementally computing the estimated cardinality (distinct elements) *distinct_col*
        where *filter_col* is equal to *filter_value*.

        @param filter_col: column to filter on.
        @param filter_value: value to filter for, i.e., df[df[filter_col] == filter_value].
        @param distinct_col: column to compute cardinality for.
        """
        super().__init__(widget)
        self.filter_col = filter_col
        self.filter_value = filter_value
        self.distinct_col = distinct_col

        # HLL for estimating cardinality. Don't modify the parameters; the autograder relies on it.
        # IMPORTANT: Please convert your data to the String type before adding to the HLL, i.e., self.hll.add(str(data))
        self.hll = HyperLogLog(p=2, seed=123456789)

        # Put any other bookkeeping class variables you need here...

    def process_slice(self, df_slice: pd.DataFrame) -> None:
        """
        Update the running filtered cardinality with a dataframe slice.
        """
        filtered_df = df_slice[df_slice[self.filter_col] == self.filter_value]
        dist_val = filtered_df[self.distinct_col].astype(str)
        for val in dist_val:
            self.hll.add(str(val))

        # Update the plot. The filtered cardinality should be put into a singleton list due to Plotly semantics.
        filt_card = self.hll.cardinality()
        self.update_widget([""], [filt_card])