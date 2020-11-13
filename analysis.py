# file: analysis.py
# purpose: rank and compare dividend stocks

import numpy as np
import pandas as pd
import os


def p2f(percent):
    return float(percent.strip("%"))/100


def f2p(float_):
    return str(round(100*float_, 2))+"%"


def insert_position(position, list1, list2):
    return list1[:position] + list2 + list1[position:]


def ignore_symbols(path):
    ignore = pd.read_csv(path)
    num_col = ignore.columns.get_loc("Market") + 1
    ignore = ignore.iloc[:, 0:num_col]
    nan_ = "nan"
    columns = ignore.columns
    ignore_list = ignore.iloc[:, 0].apply(str)
    filt = (ignore_list != nan_)
    ignore_list = ignore.loc[filt, columns[0]].to_numpy()
    for i in range(num_col-1):
        filt = (ignore.iloc[:, i+1].apply(str) != nan_)
        ignore_list = np.append(ignore_list,
                                ignore.loc[filt, columns[i+1]].to_numpy())
    return ignore_list


class Performance:

    def __init__(self, path, raw_columns, del_columns, ignore_symbols=[]):
        self.path = path
        self.ignore = ignore_symbols
        # read raw data and store it into a list
        with open(path, "r") as f:
            raw_data = [line.strip("\n") for line in f]
        # insert possible missing data
        if raw_data[0] == "52W Range":
            before = ['Symbol', 'Price']
            after = ['5D Perf', '1M Perf', '6M Perf', 'YTD Perf', '1Y Perf', '3Y Perf',
                     '3Y Total Return', '5Y Perf', '5Y Total Return', '10Y Perf', '10Y Total Return']
            raw_data = insert_position(1, raw_data, after)
            raw_data = insert_position(0, raw_data, before)
        elif raw_data[0] != "52W Range":
            before = ['Symbol', 'Price', '52W Range', '5D Perf', '1M Perf', '6M Perf', 'YTD Perf', '1Y Perf', '3Y Perf',
                      '3Y Total Return', '5Y Perf', '5Y Total Return', '10Y Perf', '10Y Total Return']
            raw_data = insert_position(0, raw_data, before)
        # length of raw data
        len_data = len(raw_data)
        raw_rows = int(len_data/raw_columns)
        self.entries = raw_rows - 1
        # transform data from vector to spreadsheet-like matrix
        np_data = np.reshape(raw_data, (raw_rows, raw_columns))
        # identify column names for data
        header = np_data[0, :]
        # remove column names from data
        np_data = np.delete(np_data, 0, 0)
        # add column names to dataframes
        self.df = pd.DataFrame(np_data, columns=header)
        # set Symbol column as index
        self.df.set_index("Symbol", inplace=True)
        # delete irrelevant columns
        for _ in range(len(del_columns)):
            del self.df[del_columns[_]]
        # start analysis
        self.analysis()

    def analysis(self):
        # add 3y data columns to dataframe
        col_3y_perf = self.df.columns.get_loc("3Y Perf")
        self.df.insert(col_3y_perf+1, "3Y Div Perf", np.nan)
        self.df.insert(col_3y_perf+3, "3Y Annualized Div Perf", np.nan)
        # add 5y data columns to dataframe
        col_5y_perf = self.df.columns.get_loc("5Y Perf")
        self.df.insert(col_5y_perf+1, "5Y Div Perf", np.nan)
        self.df.insert(col_5y_perf+3, "5Y Annualized Div Perf", np.nan)
        # add 10y data columns to dataframe
        col_10y_perf = self.df.columns.get_loc("10Y Perf")
        self.df.insert(col_10y_perf+1, "10Y Div Perf", np.nan)
        self.df.insert(col_10y_perf+3, "10Y Annualized Div Perf", np.nan)
        # list of column indexes to have data convert from str percent to float
        self.col_mod = np.array([col_3y_perf, col_3y_perf+2,
                                 col_5y_perf, col_5y_perf+2,
                                 col_10y_perf, col_10y_perf+2])
        # list of column indexes to have new data computed
        self.col_add = np.array([col_3y_perf+1, col_3y_perf+3,
                                 col_5y_perf+1, col_5y_perf+3,
                                 col_10y_perf+1, col_10y_perf+3])
        # list of significant years for data
        years = np.array([3, 5, 10])
        # list that will be annualized data columns
        self.col_annual = np.array([], dtype=int)
        # transform relevant data to float
        self.p2f_data(self.col_mod)
        # for loop to iterate through added columns
        for j in range(int(len(years))):
            # NUMy div perf
            self.df.iloc[:, self.col_add[2*j]] \
                = self.df.iloc[:, self.col_mod[2*j+1]] \
                - self.df.iloc[:, self.col_mod[2*j]]
            # NUMy annualized div perf
            self.df.iloc[:, self.col_add[2*j+1]] \
                = self.df.iloc[:, self.col_add[2*j]]/years[j]
            # list of annualized data columns
            self.col_annual = np.append(self.col_annual, self.col_add[2*j+1])

        # average annualized div performance from 3y, 5y, 10y data
        ave_annual = self.df.iloc[:, self.col_annual].mean(axis=1)
        # creating average annualized div performance column
        self.df["Ave Annualized Div Perf"] = ave_annual
        # column index for average annualized div performance
        col_ave_annual_str = "Ave Annualized Div Perf"
        col_ave_annual = self.df.columns.get_loc("Ave Annualized Div Perf")
        # appending average annualized div performance column to added column list
        self.col_add = np.append(self.col_add, col_ave_annual)
        # appending average annualized div performance column to annual data column list
        self.col_annual = np.append(self.col_annual, col_ave_annual)
        # sort data by average annualized div performance
        self.df.sort_values(self.df.columns[col_ave_annual], inplace=True,
                            ascending=False)

        # number of symbols to ignore during allocation
        len_ignore = len(self.ignore)
        if len_ignore == 1:
            filt = (self.df.index != self.ignore[0])
            # add all average annualized div perf
            sum_ = self.df.loc[filt, col_ave_annual_str].sum()
            # determine allocations for stocks based on average annualized div perf
            self.df["Allocations"] = (
                1/sum_)*self.df.loc[filt, col_ave_annual_str]
        elif len_ignore > 1:
            filt = (self.df.index != self.ignore[0])
            for i in range(len_ignore-1):
                _ = (self.df.index != self.ignore[i+1])
                filt = (filt & _)
            # add all average annualized div perf
            sum_ = self.df.loc[filt, col_ave_annual_str].sum()
            # determine allocations for stocks based on average annualized div perf
            self.df["Allocations"] = (1/sum_) * \
                self.df.loc[filt, col_ave_annual_str]
        elif len_ignore == 0:
            # add all average annualized div perf
            sum_ = self.df.iloc[:, col_ave_annual].sum()
            # determine allocations for stocks based on average annualized div perf
            self.df["Allocations"] = (1/sum_)*self.df.iloc[:, col_ave_annual]
        # column index for allocations
        col_allocation = self.df.columns.get_loc("Allocations")
        # filter of real numbers in allocation column
        filt_num = (self.df.iloc[:, col_allocation].apply(np.isnan) != True)
        # round allocations to nearest interger
        round_ = self.df.loc[filt_num, "Allocations"].apply(round, args=(2,))
        self.df.loc[filt_num, "Allocations"] = round_
        # appending allocation column to added column list
        self.col_add = np.append(self.col_add, col_allocation)

    def p2f_data(self, column_list):
        for j in range(len(column_list)):
            filt = (self.df.iloc[:, column_list[j]].str.contains("%"))
            edit = self.df.loc[filt,
                               self.df.columns[column_list[j]]].apply(p2f)
            self.df.iloc[:, column_list[j]] = edit

    def f2p_data(self, column_list):
        for j in range(len(column_list)):
            filt = (self.df.iloc[:, column_list[j]].apply(np.isnan) != True)
            edit = self.df.loc[filt,
                               self.df.columns[column_list[j]]].apply(f2p)
            self.df.iloc[:, column_list[j]] = edit

    def present(self):
        self.f2p_data(self.col_mod)
        self.f2p_data(self.col_add)

    def export(self):
        export_path = self.path.split(".")[0] + ".csv"
        self.df.to_csv(export_path)


class Dividends:

    def __init__(self, path, raw_columns, del_columns, ignore_symbols=[]):
        self.path = path
        self.ignore = ignore_symbols
        # read raw data and store it into a list
        with open(path, "r") as f:
            raw_data = [line.strip("\n") for line in f]
        # insert possible missing data
        if raw_data[0] != "Symbol":
            before = ['Symbol', 'Ex-Div Date', 'Payout Date', 'Yield TTM', 'Yield FWD', '4Y Avg Yield',
                      'Div Rate TTM', 'Div Rate FWD', 'Payout Ratio', '4Y Avg Payout', 'Div Growth 3Y', 'Div Growth 5Y', 'Years of Growth']
            raw_data = insert_position(0, raw_data, before)
        # length of raw data
        len_data = len(raw_data)
        raw_rows = int(len_data/raw_columns)
        self.entries = raw_rows - 1
        # transform data from vector to spreadsheet-like matrix
        np_data = np.reshape(raw_data, (raw_rows, raw_columns))
        # identify column names for data
        header = np_data[0, :]
        # remove column names from data
        np_data = np.delete(np_data, 0, 0)
        # add column names to dataframes
        self.df = pd.DataFrame(np_data, columns=header)
        # set Symbol column as index
        self.df.set_index("Symbol", inplace=True)
        # delete irrelevant columns
        for _ in range(len(del_columns)):
            del self.df[del_columns[_]]
        # start analysis
        self.analysis()

    def analysis(self):
        # column index for 4y average yield
        col_fwd_yield = self.df.columns.get_loc("Yield FWD")
        col_yields = [col_fwd_yield-1, col_fwd_yield]
        self.col_mod = np.array(col_yields)
        self.p2f_data(self.col_mod)
        ave_yield = self.df.iloc[:, col_yields].mean(axis=1)
        self.df.insert(col_fwd_yield+1, "2Y Ave Yield", ave_yield)

        # add average dividend growth column
        col_5y_growth = self.df.columns.get_loc("Div Growth 5Y")
        self.df.insert(col_5y_growth+1, "Ave Div Growth", np.nan)
        # list of dividend growth columns
        col_growth = np.array([col_5y_growth-1, col_5y_growth], dtype=int)
        # list of column indexes to have data convert from str percent to float
        self.col_mod = np.append(self.col_mod, col_growth)
        # transform relevant data to float
        self.p2f_data(col_growth)
        # average dividend growth
        ave_growth = self.df.iloc[:, col_growth].mean(axis=1)
        # creating average dividend growth
        self.df["Ave Div Growth"] = ave_growth
        # list of column indexes to have new data computed
        self.col_add = np.array([col_5y_growth+1])

    def p2f_data(self, column_list):
        for j in range(len(column_list)):
            filt = (self.df.iloc[:, column_list[j]].str.contains("%"))
            edit = self.df.loc[filt,
                               self.df.columns[column_list[j]]].apply(p2f)
            self.df.iloc[:, column_list[j]] = edit

    def f2p_data(self, column_list):
        for j in range(len(column_list)):
            filt = (self.df.iloc[:, column_list[j]].apply(np.isnan) != True)
            edit = self.df.loc[filt,
                               self.df.columns[column_list[j]]].apply(f2p)
            self.df.iloc[:, column_list[j]] = edit

    def present(self):
        self.f2p_data(self.col_mod)
        self.f2p_data(self.col_add)

    def export(self):
        export_path = self.path.split(".")[0] + ".csv"
        self.df.to_csv(export_path)


class Holdings:

    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(path)


class Stocks:

    def __init__(self, data_list, years=[5, 10, 15, 20, 25],
                 ignore_symbols=[], sort_column=None):
        self.ignore = ignore_symbols
        self.sort = sort_column
        self.df_all = pd.concat(data_list, axis=1)
        self.years = years
        cols = ["Allocations", "Ave Annualized Div Perf", "2Y Ave Yield",
                "Ave Div Growth"]
        self.df = self.df_all.loc[:, cols]
        self.col_mod = np.arange(0, len(cols), dtype=int)
        self.analysis()

    def analysis(self):
        # column index value to 4y ave yield
        col_yield = self.df.columns.get_loc("2Y Ave Yield")
        col_yield_str = "2Y Ave Yield"
        yield_ = self.df.iloc[:, col_yield]
        div_grow = self.df.iloc[:, col_yield+1]

        self.col_add = np.array([], dtype=int)
        # yield on cost after input years
        for i in range(len(self.years)):
            yoc = yield_*((1+div_grow)**self.years[i])
            col_yoc_str = str(self.years[i]) + "Y YoC"
            self.df[col_yoc_str] = yoc
            col_yoc = self.df.columns.get_loc(col_yoc_str)
            # list of column indexes to have new data computed
            self.col_add = np.append(self.col_add, col_yoc)

        # sort data frame
        if self.sort != None:
            col_focus_str = self.sort
            col_focus = self.df.columns.get_loc(col_focus_str)
        else:
            col_focus_str = col_yield_str
            col_focus = col_yield
        # sort data by focus column
        self.df.sort_values(self.df.columns[col_focus], inplace=True,
                            ascending=False)

        # number of symbols to ignore during allocation
        len_ignore = len(self.ignore)
        col_allocation_str = "Allocations"
        if len_ignore == 1:
            filt = (self.df.index != self.ignore[0])
            # add all average annualized div perf
            sum_ = self.df.loc[filt, col_yield_str].sum()
            # determine allocations for stocks based on average annualized div perf
            self.df[col_allocation_str] = (
                1/sum_)*self.df.loc[filt, col_yield_str]
        elif len_ignore > 1:
            filt = (self.df.index != self.ignore[0])
            for i in range(len_ignore-1):
                _ = (self.df.index != self.ignore[i+1])
                filt = (filt & _)
            # add all average annualized div perf
            sum_ = self.df.loc[filt, col_yield_str].sum()
            # determine allocations for stocks based on average annualized div perf
            self.df[col_allocation_str] = (1/sum_) * \
                self.df.loc[filt, col_yield_str]
        elif len_ignore == 0:
            # add all average annualized div perf
            sum_ = self.df.iloc[:, col_focus].sum()
            # determine allocations for stocks based on average annualized div perf
            self.df[col_allocation_str] = (1/sum_)*self.df.iloc[:, col_yield]
        # column index for allocations
        col_allocation = self.df.columns.get_loc(col_allocation_str)
        # filter of real numbers in allocation column
        filt_num = (self.df.iloc[:, col_allocation].apply(np.isnan) != True)
        # round allocations to nearest interger
        round_ = self.df.loc[filt_num,
                             col_allocation_str].apply(round, args=(2,))
        self.df.loc[filt_num, col_allocation_str] = round_

        # portfolio yield
        self.portfolio_yield = self.df.loc[filt_num, col_allocation_str] \
            * self.df.loc[filt_num, col_yield_str]
        self.portfolio_yield = self.portfolio_yield.sum()

    def p2f_data(self, column_list):
        for j in range(len(column_list)):
            filt = (self.df.iloc[:, column_list[j]].str.contains("%"))
            edit = self.df.loc[filt,
                               self.df.columns[column_list[j]]].apply(p2f)
            self.df.iloc[:, column_list[j]] = edit

    def f2p_data(self, column_list):
        for j in range(len(column_list)):
            filt = (self.df.iloc[:, column_list[j]].apply(np.isnan) != True)
            edit = self.df.loc[filt,
                               self.df.columns[column_list[j]]].apply(f2p)
            self.df.iloc[:, column_list[j]] = edit

    def present(self):
        self.f2p_data(self.col_mod)
        self.f2p_data(self.col_add)
        self.portfolio_yield = f2p(self.portfolio_yield)
        print(self.df)
        print("\n===========\t===========\t===========\t===========\t===========\t\n")
        print("Average Portfolio Yield: ", self.portfolio_yield)
        print("\n===========\t===========\t===========\t===========\t===========\t\n")
        print(self.df.info())

    def export(self):
        # get current working directory
        cwd = os.getcwd()
        export_path = cwd + "/data/stocks.csv"
        self.df.to_csv(export_path)


if __name__ == "__main__":
    # set max columns to view in dataframe
    pd.set_option("display.max_columns", 11)
    # get current working directory
    cwd = os.getcwd()

    # ignore list
    path_ignore = cwd + "/data/ignore.csv"
    ignore_list = ignore_symbols(path_ignore)

    # performance info
    path_perf = cwd + "/data/performance.txt"
    columns_perf = 14
    del_columns_perf = ["Price", "52W Range"]
    # performance analysis
    perf = Performance(path_perf, columns_perf, del_columns_perf, ignore_list)

    # dividend info
    path_div = cwd + "/data/dividends.txt"
    columns_div = 13
    del_columns_div = ["Ex-Div Date", "Payout Date"]
    allocation_ignore_div = 1
    # dividends analysis
    div = Dividends(path_div, columns_div, del_columns_div, ignore_list)

    # holding info
    path_hold = cwd + "/personal/m1.csv"
    hold = Holdings(path_hold)
    print(hold.df)

    data = [perf.df, div.df]
    cols = ["Ave Annualized Div Perf", "2Y Ave Yield", "Ave Div Growth", ]
    # stocks = Stocks(data, ignore_symbols=ignore_list, sort_column=cols[1])
    stocks = Stocks(data, ignore_symbols=ignore_list)
    # stocks.present()
    # stocks.export()
