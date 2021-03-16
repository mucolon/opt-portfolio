#!/usr/local/bin/python3

# Setup code
import numpy as np
import pandas as pd
from pandas_datareader import data as web
import os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import scipy.optimize as opt
from datetime import datetime
import pytz
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns


def p2f(percent):
    '''Convert string percent to a float number
    '''
    return float(percent.strip("%")) / 100


def f2p(float_):
    '''Convert float number to a string percent
    '''
    return str(round(100 * float_, 2)) + "%"


def dollar2f(dollar):
    '''Convert string dollar to float number
    '''
    return float(dollar.strip("$"))


def f2dollar(float_, drop_cents=False):
    '''Convert float number to string dollar
    '''
    dollar = "$" + str(round(float_, 2))
    length = len(dollar)
    decimal = dollar.index(".")
    if (length - decimal) == 2:
        dollar = dollar + "0"
    if drop_cents:
        dollar = dollar.split(".")[0]
    return dollar


def insert_position(position, list1, list2):
    '''Insert list2 into position of list1
    '''
    return list1[:position] + list2 + list1[position:]


def remove_duplicates(list_):
    '''Remove duplicate objects from within input list
    '''
    _ = []
    [_.append(i) for i in list_ if i not in _]
    return _


def performance_data(path, raw_columns):
    '''Convert performance data text file to a dataframe
    '''
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
    raw_rows = int(len_data / raw_columns)
    print(raw_rows, "\n", raw_columns, "\n", raw_data)
    # transform data from vector to spreadsheet-like matrix
    np_data = np.reshape(raw_data, (raw_rows, raw_columns))
    # identify column names for data
    header = np_data[0, :]
    # remove column names from data
    np_data = np.delete(np_data, 0, 0)
    # add column names to dataframes
    df = pd.DataFrame(np_data, columns=header)
    # set Symbol column as index
    df.set_index("Symbol", inplace=True)
    df = df.rename_axis(None)
    return df


def dividends_data(path, raw_columns):
    '''Convert dividend data text file to a dataframe
    '''
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
    raw_rows = int(len_data / raw_columns)
    # transform data from vector to spreadsheet-like matrix
    np_data = np.reshape(raw_data, (raw_rows, raw_columns))
    # identify column names for data
    header = np_data[0, :]
    # remove column names from data
    np_data = np.delete(np_data, 0, 0)
    # add column names to dataframes
    df = pd.DataFrame(np_data, columns=header)
    # set Symbol column as index
    df.set_index("Symbol", inplace=True)
    df = df.rename_axis(None)
    return df


def excel_data(path, sheet_names):
    '''Convert excel spreadsheet to dataframe
    '''
    dic = pd.read_excel(path,
                        sheet_name=sheet_names, engine="openpyxl")
    df = pd.concat([pd.DataFrame.from_dict(dic[i])
                    for i in sheet_names], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df.set_index("Symbol", inplace=True)
    df = df.rename_axis(None)
    df.replace("-", np.nan, inplace=True)
    return df


def print_divider(num_symbol):
    '''
    '''
    _ = "*"
    _ = str(num_symbol * _)
    tab = "\t"
    tab2 = "\t\t"
    _ = tab + _ + tab2 + _ + tab2 + _ + tab2 + _
    print(_)


def exp_func(x, a, b, c):
    '''Exponential function with parameters for constants. Function: a*exp(b*x)+c. Input: x can be an int or list
    '''
    try:
        len(x)
    except TypeError:
        return (a * np.exp(b * x) + c)
    return [(a * np.exp(b * i) + c) for i in x]


def fit_data(func, xdata, ydata, return_r_squared=False):
    '''
    '''
    popt, pcov = opt.curve_fit(func, xdata, ydata)
    fit_data = func(xdata, *popt)
    fit_data = np.array(fit_data)
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    if return_r_squared is False:
        residuals = fit_data - ydata
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((ydata - np.mean(ydata))**2)
        r_squared = 1 - (ss_res / ss_tot)
        return fit_data, popt, r_squared
    else:
        return fit_data, popt


class Watchlist:
    '''Class filters Stocks.xlsx to a watchlist dataframe that passes the filter_poor() method
    '''

    def __init__(self, dataframe, percent_columns, dollar_columns, round_columns, cwd=None):
        self.df = dataframe
        self.col_per = percent_columns
        self.col_dol = dollar_columns
        self.col_round = round_columns
        self.cwd = cwd
        self.df.rename(columns={"4Y Avg Yield": "Ave Yield"}, inplace=True)

    def yield_(self, str_, str_ave):
        '''Add yield data column, will be FWD yield if present or TTM yield if FWD is missing
        '''
        self.str_yield = str_
        self.str_yield_ave = str_ave
        col_fwd_yield = self.df.columns.get_loc("Yield FWD")
        col_yields = [col_fwd_yield - 1, col_fwd_yield]
        port_yield = self.df.iloc[:, col_yields].mean(axis=1)
        self.fwd_yield = 2 * port_yield - self.df.iloc[:, col_fwd_yield - 1]
        self.df.insert(col_fwd_yield + 1, str_, self.fwd_yield)
        self.col_per = np.append(self.col_per, str_)

    def div_rate(self, str_):
        '''Add div rate data column, will be FWD div rate if present or TTM div rate if FWD is missing
        '''
        self.str_div_rate = str_
        col_fwd_rate = self.df.columns.get_loc("Div Rate FWD")
        col_rates = [col_fwd_rate - 1, col_fwd_rate]
        ave_rate = self.df.iloc[:, col_rates].mean(axis=1)
        fwd_rate = 2 * ave_rate - self.df.iloc[:, col_fwd_rate - 1]
        self.df.insert(col_fwd_rate + 1, str_, fwd_rate)
        self.col_dol = np.append(self.col_dol, str_)

    def ave_div_perf(self, str_):
        '''Add ave div perf data column
        '''
        self.str_ave_perf = str_
        str_3y_perf = "3Y Perf"
        str_3y_div_perf = "3Y Div Perf"
        str_3y_total = "3Y Total Return"
        str_5y_perf = "5Y Perf"
        str_5y_div_perf = "5Y Div Perf"
        str_5y_total = "5Y Total Return"
        str_10y_perf = "10Y Perf"
        str_10y_div_perf = "10Y Div Perf"
        str_10y_total = "10Y Total Return"
        col_3y_total = self.df.columns.get_loc(str_3y_total)
        col_5y_total = self.df.columns.get_loc(str_5y_total)
        col_10y_total = self.df.columns.get_loc(str_10y_total)
        ave_3y_div_perf = (self.df[str_3y_total] - self.df[str_3y_perf]) / 3
        ave_5y_div_perf = (self.df[str_5y_total] - self.df[str_5y_perf]) / 5
        ave_10y_div_perf = (self.df[str_10y_total] -
                            self.df[str_10y_perf]) / 10
        self.df.insert(col_3y_total + 1, str_3y_div_perf, ave_3y_div_perf)
        self.df.insert(col_5y_total + 1, str_5y_div_perf, ave_5y_div_perf)
        self.df.insert(col_10y_total + 1, str_10y_div_perf, ave_10y_div_perf)
        col_div_perf = [str_3y_div_perf, str_5y_div_perf, str_10y_div_perf]
        self.ave_perf = self.df.loc[:, col_div_perf].mean(axis=1)
        self.df.insert(col_10y_total + 2, str_, self.ave_perf)
        self.col_per = np.append(self.col_per, col_div_perf)
        self.col_per = np.append(self.col_per, str_)

    def ave_div_growth(self, str_):
        '''Add ave div growth data column
        '''
        self.str_div_growth = str_
        col_5y_growth = self.df.columns.get_loc("Div Growth 5Y")
        col_growth = [col_5y_growth - 1, col_5y_growth]
        str_columns = self.df.columns.tolist()
        str_growth = str_columns[col_growth[0]:col_growth[-1] + 1]
        for i in str_growth:
            filt = (self.df.loc[:, i].apply(type) == str)
            self.df.loc[filt, i] = self.df.loc[filt, i].apply(float) / 100
        self.ave_growth = self.df.iloc[:, col_growth].mean(axis=1)
        self.df.insert(col_5y_growth + 1, str_, self.ave_growth)
        self.col_per = np.append(self.col_per, str_)

    def yoc_years(self, years):
        '''Add projected yoc data columns
        '''
        col_ave_grow = self.df.columns.get_loc(self.str_div_growth)
        _ = 1
        for i in years:
            yoc = self.fwd_yield * ((1 + self.ave_growth)**i)
            str_ = str(i) + "Y YoC"
            self.df.insert(col_ave_grow + _, str_, yoc)
            self.col_per = np.append(self.col_per, str_)
            _ += 1

    def pe_ratio(self, str_):
        '''Add P/E ratio data column, will be FWD P/E if present or TTM P/E if FWD P/E is missing
        '''
        self.str_pe = str_
        col_fwd_pe = self.df.columns.get_loc("P/E FWD")
        col_pes = [col_fwd_pe - 1, col_fwd_pe]
        ave_pe = self.df.iloc[:, col_pes].mean(axis=1)
        filt = (ave_pe == self.df.iloc[:, col_fwd_pe])
        fwd_pe = 2 * ave_pe - self.df.iloc[:, col_fwd_pe - 1]
        fwd_pe[filt] = ave_pe
        self.df.insert(col_fwd_pe + 1, str_, fwd_pe)
        self.col_round = np.append(self.col_round, str_)

    def filter_poor(self, str_):
        '''Remove symbols based on poor yield, YoC, div growth, payout ratio, 3y, 5y, and 10y performances
        '''
        str_yoc_year = str_
        self.str_yoc_year = str_yoc_year
        str_yield = self.str_yield
        str_years_growth = "Years of Growth"
        str_payout = "Payout Ratio"
        str_div_growth = self.str_div_growth
        str_3y_div_growth = "Div Growth 3Y"
        str_5y_div_growth = "Div Growth 5Y"
        str_3y_perf = "3Y Perf"
        str_3y_total = "3Y Total Return"
        str_5y_perf = "5Y Perf"
        str_5y_total = "5Y Total Return"
        str_10y_perf = "10Y Perf"
        str_10y_total = "10Y Total Return"
        str_pe = self.str_pe
        str_schd = "SCHD"
        threshold_yoc = self.df.loc[str_schd, str_yoc_year] * 1.1
        threshold_yield = self.df.loc[str_schd, str_yield] * 1.1
        threshold_filt = ((self.df[str_yield] < threshold_yield) &
                          (self.df[str_yoc_year] < threshold_yoc))
        threshold_filt[str_schd] = False
        filt = (threshold_filt | (self.df[str_pe] < 0) | (self.df[str_pe] > 100) |
                (self.df[str_years_growth].str[0] == "0") | (self.df[str_payout] > 0.95) |
                (pd.isnull(self.df[str_yield])) | (pd.isnull(self.df[str_3y_div_growth])) |
                (self.df[str_3y_div_growth] < 0) | (self.df[str_5y_div_growth] < 0) |
                # dividend growth rate filter set by O
                (self.df[str_div_growth] < 0.034) |
                # 3y performance filter set by ABBV
                (self.df[str_3y_perf] < -0.2) | (self.df[str_3y_total] < -0.03) |
                (self.df[str_5y_perf] < 0) | (self.df[str_5y_total] < 0) |
                (self.df[str_10y_perf] < 0) | (self.df[str_10y_total] < 0))
        remove_script = self.df.loc[filt].index.values.tolist()
        self.remove_script = remove_script

    def update_ignore_list(self, path_ignore, exceptions=None):
        '''Update ignore csv based on symbols found after filter_poor() method
        '''
        if exceptions is None:
            exceptions = []
        self.exceptions = exceptions
        remove_script = self.remove_script  # symbols that didn't pass filter_poor()
        ignore_df = pd.read_csv(path_ignore)
        col_index_portfolio = ignore_df.columns.get_loc("Portfolio")
        index_portfolio = ignore_df["Portfolio"].dropna().tolist()
        cols_ignore = ignore_df.columns.values.tolist()
        col_script = cols_ignore[0]
        cols_qual = cols_ignore[1:col_index_portfolio]
        cols_qual_poor = cols_qual[0:2]
        cols_qual_ave = cols_qual[2:]
        # list of lists of all symbols within qual poor cols
        _poor = [ignore_df[i].dropna(
            how="all").values.tolist() for i in cols_qual_poor]
        # list of lists of all symbols within qual ave cols
        _ave = [ignore_df[i].dropna(
            how="all").values.tolist() for i in cols_qual_ave]
        # list of all symbols within qual poor cols
        qual_poor = [item for sublist in _poor for item in sublist]
        # list of all symbols within qual ave cols
        qual_ave = [item for sublist in _ave for item in sublist]
        # list of all symbols within qual cols
        remove_qual = qual_poor + qual_ave
        self.remove_qual = remove_qual
        self.qual_poor = qual_poor
        self.qual_ave = qual_ave
        entries = self.df.index.tolist()
        # compile list of symbols that should be removed from entries b/c of poor qual details
        remove_qual_poor = [i for i in qual_poor if i in entries]
        remove_qual_poor = [i for i in remove_qual_poor if i not in exceptions]
        self.remove_qual_poor = sorted(remove_qual_poor)
        # compile list of symbols that should be added to entries b/c of ave qual details
        add_qual_ave = [i for i in qual_ave if i not in entries]
        self.add_qual_ave = sorted(add_qual_ave)
        # remove symbols if not in stock spreadsheet
        remove_qual = [i for i in remove_qual if i in entries]
        # remove symbols if in exceptions list
        remove_qual = [i for i in remove_qual if i not in exceptions]
        self.df.drop(remove_qual, inplace=True)
        entries = self.df.index.tolist()
        exceptions = exceptions + index_portfolio
        exceptions = remove_duplicates(exceptions)
        if remove_script != []:
            # symbols that didn't pass filter_poor() but are part of dataframe
            self.script_override = [
                i for i in index_portfolio if i in remove_script]
            # remove symbols if in exceptions list
            remove_edited = [i for i in remove_script if i not in exceptions]
            # remove symbols if not in stock spreadsheet
            remove_edited = [i for i in remove_edited if i in entries]
            self.df.drop(remove_edited, inplace=True)
            script_df = pd.concat([pd.DataFrame([i], columns=[col_script])
                                   for i in remove_edited], ignore_index=True)
            ignore_df.drop(columns=col_script, inplace=True)
            ignore_df = pd.concat([script_df, ignore_df], axis=1)
        ignore_df.to_csv(path_ignore, index=False)
        self.index_portfolio = index_portfolio
        self.entries = self.df.index.tolist()

    def update_watchlist(self, path_list):
        '''Update watchlist text file based on symbols that passed filter_poor() method
        '''
        entries = self.df.index.tolist()
        # remove symbols that have been qualitatively removed
        entries = [i for i in entries if i not in self.remove_qual]
        # remove symbols that didn't pass filter_poor()
        remove_script = [i for i in self.remove_script
                         if i not in self.script_override]
        entries = [i for i in entries if i not in remove_script]
        # entries = sorted(entries)
        with open(path_list, "w") as f:
            f.write(',\n'.join(entries))

    def sort(self, column, ascending=False, return_dataframe=False, input_dataframe=None):
        '''Sort dataframe based on specified column. If return_dataframe is True, a new modified dataframe will be returned. If input_dataframe is not None, input_dataframe will be used to modify
        '''
        if return_dataframe:
            if input_dataframe is None:
                new_df = self.df.copy()
            else:
                new_df = input_dataframe
            new_df.sort_values(column, inplace=True, ascending=ascending)
            return new_df
        else:
            self.df.sort_values(column, inplace=True, ascending=ascending)

    def portfolio_mark(self, str_):
        '''Mark symbols that are part of portfolio
        '''
        self.str_mark = str_
        self.df.loc[self.index_portfolio, str_] = u'\u2713'

    def p2f_data(self, return_dataframe=False, input_dataframe=None):
        '''Convert string percent data columns to float number data columns. If return_dataframe is True, a new modified dataframe will be returned. If input_dataframe is not None, input_dataframe will be used to modify
        '''
        if return_dataframe:
            if input_dataframe is None:
                new_df = self.df.copy()
            else:
                new_df = input_dataframe
            for i in self.col_per:
                filt = (new_df.loc[:, i].str.contains("%"))
                edit = new_df.loc[filt, i].apply(p2f)
                new_df.loc[:, i] = edit
            return new_df
        else:
            for i in self.col_per:
                try:
                    filt = (self.df.loc[:, i].str.contains("%"))
                except AttributeError:
                    continue
                edit = self.df.loc[filt, i].apply(p2f)
                self.df.loc[:, i] = edit

    def f2p_data(self, return_dataframe=False, input_dataframe=None):
        '''Convert float number data columns to string percent data columns. If return_dataframe is True, a new modified dataframe will be returned. If input_dataframe is not None, input_dataframe will be used to modify
        '''
        if return_dataframe:
            if input_dataframe is None:
                new_df = self.df.copy()
            else:
                new_df = input_dataframe
            for i in self.col_per:
                filt = (new_df.loc[:, i].apply(pd.isnull) != True)
                edit = new_df.loc[filt, i].apply(f2p)
                new_df.loc[:, i] = edit
            return new_df
        else:
            for i in self.col_per:
                filt = (self.df.loc[:, i].apply(pd.isnull) != True)
                edit = self.df.loc[filt, i].apply(f2p)
                self.df.loc[:, i] = edit

    def dollar2f_data(self, return_dataframe=False, input_dataframe=None):
        '''Convert string dollar data columns to float number data columns. If return_dataframe is True, a new modified dataframe will be returned. If input_dataframe is not None, input_dataframe will be used to modify
        '''
        if return_dataframe:
            if input_dataframe is None:
                new_df = self.df.copy()
            else:
                new_df = input_dataframe
            for i in self.col_dol:
                filt = (~new_df.loc[:, i].str.contains("-"))
                edit = new_df.loc[filt, i].apply(dollar2f)
                new_df.loc[:, i] = edit
            return new_df
        else:
            for i in self.col_dol:
                try:
                    filt = (~self.df.loc[:, i].str.contains("-"))
                except AttributeError:
                    continue
                edit = self.df.loc[filt, i].apply(dollar2f)
                self.df.loc[:, i] = edit

    def f2dollar_data(self, return_dataframe=False, input_dataframe=None):
        '''Convert float number data columns to string percent data columns. If return_dataframe is True, a new modified dataframe will be returned. If input_dataframe is not None, input_dataframe will be used to modify
        '''
        if return_dataframe:
            if input_dataframe is None:
                new_df = self.df.copy()
            else:
                new_df = input_dataframe
            for i in self.col_dol:
                filt = (new_df.loc[:, i].apply(pd.isnull) != True)
                edit = new_df.loc[filt, i].apply(f2dollar)
                new_df.loc[:, i] = edit
            return new_df
        else:
            for i in self.col_dol:
                filt = (self.df.loc[:, i].apply(pd.isnull) != True)
                edit = self.df.loc[filt, i].apply(f2dollar)
                self.df.loc[:, i] = edit

    def round_data(self, place=2, return_dataframe=False, input_dataframe=None):
        '''Round data columns to decimal place argument. If return_dataframe is True, a new modified dataframe will be returned. If input_dataframe is not None, input_dataframe will be used to modify
        '''
        if return_dataframe:
            if input_dataframe is None:
                new_df = self.df.copy()
            else:
                new_df = input_dataframe
            for i in self.col_round:
                filt = (new_df.loc[:, i].apply(pd.isnull) != True)
                edit = new_df.loc[filt, i].apply(round, args=(place,))
                new_df.loc[:, i] = edit
            return new_df
        else:
            for i in self.col_round:
                filt = (self.df.loc[:, i].apply(pd.isnull) != True)
                edit = self.df.loc[filt, i].apply(round, args=(place,))
                self.df.loc[:, i] = edit

    def cleanup_data(self, return_dataframe=False, place=2):
        '''Convert data to better readable strings. If return_dataframe is True, a new modified dataframe will be returned
        '''
        if return_dataframe:
            new_df = self.f2p_data(return_dataframe=return_dataframe)
            new_df = self.f2dollar_data(return_dataframe=return_dataframe,
                                        input_dataframe=new_df)
            new_df = self.round_data(place=place, return_dataframe=return_dataframe,
                                     input_dataframe=new_df)
            return new_df
        else:
            self.f2p_data()
            self.f2dollar_data()
            self.round_data()

    def print_terminal(self, columns, sort_column, ascending=False, num_symbol=16):
        '''Print specificed dataframe columns onto terminal
        '''
        print_divider(num_symbol)
        self.sort(sort_column, ascending=ascending)
        df = self.cleanup_data(return_dataframe=True)
        print(df.loc[:, columns])
        print("Index Length:", len(df.index))

    def export_csv(self, file_name, columns, sort_column, ascending=False):
        '''Export specificed dataframe columns to a csv
        '''
        self.sort(sort_column, ascending=ascending)
        df = self.cleanup_data(return_dataframe=True)
        export_path = self.cwd + "/data/" + file_name + ".csv"
        df.loc[:, columns].to_csv(export_path)

    def graph_yield_yoc(self, omit_symbols=None):
        '''
        '''
        if omit_symbols is None:
            omit_symbols = []
        fig, ax = plt.subplots()
        for i in self.entries:
            if i in omit_symbols:
                continue
            ax.plot(self.df.loc[i, self.str_yield],
                    self.df.loc[i, self.str_yoc_year], ".", label=i)
        str_title = "Symbol: " + self.str_yoc_year + " vs. Yield"
        ax.set_title(str_title)
        ax.legend(loc="best", ncol=3)
        plt.show()


class Portfolio(Watchlist):
    '''Class highlights portfolio performance
    '''

    def __init__(self, watchlist_obj):
        watch = watchlist_obj
        self.exceptions = watch.exceptions
        self.remove_qual_poor = watch.remove_qual_poor
        self.add_qual_ave = watch.add_qual_ave
        self.script_override = watch.script_override
        self.index_portfolio = watch.index_portfolio
        self.str_yield = watch.str_yield
        self.str_div_rate = watch.str_div_rate
        index_ = watch.index_portfolio + watch.exceptions
        index_ = remove_duplicates(index_)
        self.df = watch.df.loc[index_, :]
        self.df = self.df.rename_axis(None)
        self.col_per = watch.col_per
        self.col_dol = watch.col_dol
        self.col_round = watch.col_round
        self.cwd = watch.cwd

    def m1_import(self, path_csv, percent_columns, dollar_columns):
        '''Import M1 csv file for portfolio analysis
        '''
        m1_df = pd.read_csv(path_csv)
        m1_df.set_index("Ticker", inplace=True)
        m1_df = m1_df.rename_axis(None)
        m1_df.rename(columns={"Avg. Price": "Ave Price"}, inplace=True)
        try:
            self.df = self.df.loc[m1_df.index.tolist(), :]
        except KeyError:
            m1_index = m1_df.index.tolist()
            df_index = self.df.index.tolist()
            error_index = [i for i in m1_index if i not in df_index]
            print("\nUPDATE EXCEPTIONS LIST: ")
            [print(i) for i in error_index]
        self.df = pd.concat([self.df, m1_df], axis=1)
        self.col_per = np.append(self.col_per, percent_columns)
        self.col_dol = np.append(self.col_dol, dollar_columns)
        self.p2f_data()
        self.dollar2f_data()

    def div_rate(self, str_annual, str_month):
        '''Add portfolio annual div rate data column based on shares, calculate total portfolio div rate based on shares
        '''
        self.str_annual_div = str_annual
        self.str_month_div = str_month
        div_annual = self.df[self.str_div_rate] * self.df["Shares"]
        div_month = div_annual / 12
        self.df[str_annual] = div_annual
        self.df[str_month] = div_month
        self.port_annual_div = self.df[str_annual].sum()
        self.port_month_div = self.df[str_month].sum()
        self.col_dol = np.append(self.col_dol, [str_annual, str_month])

    def current_allocation(self, str_):
        '''Add portfolio current allocation data amount based on market value of shares
        '''
        self.str_cur_allocate = str_
        str_value = "Value"
        total_invest = self.df[str_value].sum()
        allocation = self.df[str_value] / total_invest
        self.df[str_] = allocation
        self.col_per = np.append(self.col_per, str_)

    def yoc(self, str_):
        '''Add portfolio YoC data column based on cost basis
        '''
        self.str_yoc = str_
        yoc = self.df[self.str_div_rate] / self.df["Ave Price"]
        self.df[str_] = yoc
        self.col_per = np.append(self.col_per, str_)

    def yield_growth(self, str_):
        '''Add portfolio div growth data column
        '''
        self.str_yield_growth = str_
        yield_growth = (self.df[self.str_yoc] / self.df[self.str_yield]) - 1
        self.df[str_] = yield_growth
        self.col_per = np.append(self.col_per, str_)

    def update_portfolio_list(self, path_list, sort_column):
        '''Update portfolio list based on ignore csv portfolio column
        '''
        port = self.sort(sort_column, return_dataframe=True,
                         input_dataframe=self.df.loc[self.index_portfolio, :])
        list_port = port.index.tolist()
        with open(path_list, "w") as f:
            f.write(',\n'.join(list_port))

    def calculate_summary(self):
        '''Calculate portfolio average yield, yoc, yield growth, and performance
        '''
        self.port_yield = sum(
            self.df[self.str_cur_allocate] * self.df[self.str_yield])
        self.port_yoc = sum(
            self.df[self.str_cur_allocate] * self.df[self.str_yoc])
        self.port_yield_growth = sum(
            self.df[self.str_cur_allocate] * self.df[self.str_yield_growth])
        self.port_value = self.df["Value"].sum()
        self.port_cost = self.df["Cost Basis"].sum()
        self.port_perf = (self.port_value / self.port_cost) - 1

    def optimize(self, import_data=True):
        '''
        '''
        len_port = len(self.index_portfolio)
        weights = np.array([1 / len_port] * len_port)
        # print(self.index_portfolio)
        # print(self.df.index)
        # print(self.df[self.str_cur_allocate], "\n")
        path = self.cwd + "/data/price-data.csv"
        if import_data:
            price_df = pd.read_csv(path)
            price_df.set_index("Date", inplace=True)
        else:
            # start date based on BLOK
            date_start = "2018-01-25"
            today = datetime.today().strftime('%Y-%m-%d')
            price_df = pd.DataFrame()
            for i in self.index_portfolio:
                price_df[i] = web.DataReader(i, data_source='yahoo',
                                             start=date_start, end=today)['Adj Close']
            price_df.to_csv(path)

        trading_days = 252.75
        returns = price_df.pct_change()
        cov_matrix_annual = returns.cov() * trading_days
        port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
        port_volatility = np.sqrt(port_variance)
        # sar = simple annual return
        port_sar = np.sum(returns.mean() * weights) * trading_days
        percent_var = str(round(port_variance, 2) * 100) + '%'
        percent_vols = str(round(port_volatility, 2) * 100) + '%'
        percent_ret = str(round(port_sar, 2) * 100) + '%'
        print("Expected annual return : ", percent_ret)
        print('Annual volatility/standard deviation/risk : ', percent_vols)
        print('Annual variance : ', percent_var, "\n")
        # returns.mean() * 252
        mu = expected_returns.mean_historical_return(price_df)
        # Get the sample covariance matrix
        S = risk_models.sample_cov(price_df)
        ef = EfficientFrontier(mu, S)
        # Maximize the Sharpe ratio, and get the raw weights
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        print(cleaned_weights, "\n")
        ef.portfolio_performance(verbose=True)

    def print_summary(self, columns, sort_column, ascending=False, num_symbol=16):
        '''Print specificed dataframe columns and portfolio summary onto terminal
        '''
        port_value = f2dollar(self.port_value)
        port_cost = f2dollar(self.port_cost)
        port_perf = f2p(self.port_perf)
        port_month_div = f2dollar(self.port_month_div)
        port_annual_div = f2dollar(self.port_annual_div)
        port_yield = f2p(self.port_yield)
        port_yoc = f2p(self.port_yoc)
        port_yield_growth = f2p(self.port_yield_growth)
        script_override = str(', '.join(self.script_override))
        remove_qual_poor = str(', '.join(self.remove_qual_poor))
        add_qual_ave = str(', '.join(self.add_qual_ave))
        print_divider(num_symbol)
        self.sort(sort_column, ascending=ascending)
        df = self.cleanup_data(return_dataframe=True)
        now = datetime.now(tz=pytz.timezone('US/Pacific'))
        now = now.strftime("%b-%d-%y %H:%M:%S")
        print(df.loc[:, columns])
        print("Portfolio Value:\t\t", port_value)
        print("Portfolio Cost Basis:\t\t", port_cost)
        print("Portfolio Performance:\t\t", port_perf)
        print("Portfolio Monthly Dividends:\t", port_month_div)
        print("Portfolio Annual Dividends:\t", port_annual_div)
        print("Portfolio Yield:\t\t", port_yield)
        print("Portfolio YoC:\t\t\t", port_yoc)
        print("Portfolio Yield Growth:\t\t", port_yield_growth)
        print("Warning Symbols:\t\t", script_override)
        print("Remove Symbols:\t\t\t", remove_qual_poor)
        print("Add Symbols:\t\t\t", add_qual_ave)
        print("Date/Time:\t\t\t", now)

    def graph_history(self, path_history):
        '''
        '''
        hist_df = pd.read_csv(path_history)
        cols = hist_df.columns.tolist()
        data_date = pd.DatetimeIndex(hist_df[cols[0]])
        months = hist_df.index.tolist()
        data_value = hist_df[cols[1]]
        data_income = hist_df[cols[2]]
        hist_df[cols[0]] = data_date
        # list of years
        data_year = []
        [data_year.append(i) for i in data_date.year.tolist()
         if i not in data_year]
        # list of yearly income
        year_income = []
        for i in data_year:
            filt = (i == data_date.year)
            year_income.append(data_income.loc[filt].sum())
        # list of cumulative sum of monthly income
        cumsum_income = data_income.cumsum()
        fit_cumsum, popt_cumsum, r2_cumsum = fit_data(exp_func, months,
                                                      cumsum_income)
        # model monthly income based on curve fit of cumsum_income
        future_month0 = months[-1] + 1
        future_months = np.arange(future_month0, future_month0 + 12)
        future_months = np.concatenate([months, future_months])
        future_income = exp_func(future_months, *popt_cumsum)
        for i in range(future_months[-1], 0, -1):
            future_income[i] = future_income[i] - future_income[i - 1]
        future_date = data_date + pd.DateOffset(months=12)
        future_date = future_date[-12:]
        future_date = data_date.union(future_date)
        # curve fit monthly income
        fit_income, popt_income = fit_data(exp_func, future_months, future_income,
                                           return_r_squared=True)
        # initialize graph
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        li_green = "#6BA755"
        li_grey = "#D7D7D7"
        # portfolio value plot
        ax1.fill_between(data_date, data_value, color=li_green)
        ax1.set_title("Portfolio Value")
        # portfolio monthly income plot
        ax2.plot(data_date, data_income, ".-", label="Raw")
        ax2.plot(future_date, future_income, ".-", label="Forcast",
                 color=li_green)
        # ax2.plot(future_date, fit_income, ".-", label="Fit", color=li_green)
        str_fit_eqn = '\n'.join(("Fit Equation:",
                                 r"%.2f$e^{%.2fx}$%.2f" % (
                                     popt_income[0], popt_income[1], popt_income[2])))
        props = dict(boxstyle='round', facecolor="white", edgecolor=li_grey)
        anchored_text = AnchoredText(str_fit_eqn, loc=4, frameon=False,
                                     prop=dict(bbox=props))
        ax2.add_artist(anchored_text)
        ax2.set_title("Monthly Income")
        ax2.legend(loc="best")
        # portfolio cumsum income plot
        ax3.plot(data_date, cumsum_income, ".-", label="Raw")
        ax3.plot(data_date, fit_cumsum, label="Fit", color=li_green)
        str_fit_eqn = '\n'.join(("Fit Equation:",
                                 r"%.2f$e^{%.2fx}$%.2f" % (
                                     popt_cumsum[0], popt_cumsum[1], popt_cumsum[2]),
                                 r"$R^2$= %.2f" % (r2_cumsum,)))
        props = dict(boxstyle='round', facecolor="white", edgecolor=li_grey)
        anchored_text = AnchoredText(str_fit_eqn, loc=4, frameon=False,
                                     prop=dict(bbox=props))
        ax3.add_artist(anchored_text)
        ax3.set_title("Cumulative Sum Income")
        ax3.legend(loc="best")
        plt.show()


if __name__ == "__main__":
    # file paths
    cwd = os.getcwd()
    path_ignore = cwd + "/data/ignore.csv"
    path_list = cwd + "/data/watchlist.txt"
    path_portfolio = cwd + "/data/portfolio.txt"
    path_excel = cwd + "/data/Stocks.xlsx"
    path_m1 = cwd + "/personal/m1.csv"
    path_history = cwd + "/personal/history.csv"

    # system commands
    os.system("mv -f ~/Downloads/Stocks.xlsx " + path_excel + " 2>/dev/null")
    os.system("mv -f ~/Downloads/m1.csv " + path_m1 + " 2>/dev/null")

    # data constants
    percent_columns_perf = ["5D Perf", "1M Perf", "6M Perf", "YTD Perf", "1Y Perf", "3Y Perf",
                            "3Y Total Return", "5Y Perf", "5Y Total Return", "10Y Perf", "10Y Total Return"]
    percent_columns_div = ["Yield TTM", "Yield FWD", "Ave Yield",
                           "Payout Ratio", "4Y Avg Payout", "Div Growth 3Y", "Div Growth 5Y"]
    percent_columns_m1 = ["Unrealized Gain %"]
    dollar_columns_div = ["Div Rate TTM", "Div Rate FWD"]
    dollar_columns_m1 = ["Ave Price", "Cost Basis", "Unrealized Gain", "Value"]
    round_columns_value = ["P/E TTM", "P/E FWD"]
    percent_columns = percent_columns_perf + percent_columns_div
    dollar_columns = dollar_columns_div
    round_columns = round_columns_value

    # import data from excel file
    sheet_names = ["Performance", "Dividends", "Value", "Growth"]
    df = excel_data(path_excel, sheet_names)

    # data analysis constants
    num_symbol = 5
    str_yield = "Yield"
    str_yield_ave = "Ave Yield"
    str_div_rate = "Div Rate"
    str_div_perf = "Ave Div Perf"
    str_div_growth = "Ave Div Grow"
    str_pe = "P/E"
    str_port = "Port"
    str_month_div = "Monthly Div"
    str_annual_div = "Annual Div"
    str_cur_allocate = "Cur Allocate"
    str_yoc = "YoC"
    str_yield_growth = "Div Grow"
    str_value = "Value"
    years = [5, 10, 15, 20]
    yoc_year = years[-1]
    str_yoc_year = str(yoc_year) + "Y YoC"
    export_columns = [str_yield, str_yield_ave, str_div_perf, str_div_growth,
                      str_yoc_year, str_pe, str_port]
    m1_export_columns = [str_cur_allocate, "Shares", "Ave Price", "Cost Basis",
                         str_value, "Unrealized Gain %", str_month_div, str_annual_div, str_yield_ave, str_yield,
                         str_yoc, str_yield_growth]

    # pandas options
    pd.set_option("display.max_columns", len(export_columns))
    # pd.set_option("display.max_rows", None)

    # exceptions = ["JEPI", "BLOK"]
    exceptions = []

    # start data analysis to filter stocks to a singular watchlist
    watch = Watchlist(df, percent_columns, dollar_columns,
                      round_columns, cwd=cwd)
    watch.yield_(str_yield, str_yield_ave)
    watch.div_rate(str_div_rate)
    watch.ave_div_perf(str_div_perf)
    watch.ave_div_growth(str_div_growth)
    watch.yoc_years(years)
    watch.pe_ratio(str_pe)
    watch.filter_poor(str_yoc_year)
    watch.sort(str_yield_ave)
    watch.update_ignore_list(path_ignore, exceptions=exceptions)
    watch.update_watchlist(path_list)
    watch.portfolio_mark(str_port)
    watch.export_csv("watchlist", export_columns, str_yield_ave)

    # start data analysis to highlight portfolio performance
    port = Portfolio(watch)
    port.m1_import(path_m1, percent_columns_m1, dollar_columns_m1)
    port.div_rate(str_annual_div, str_month_div)
    port.current_allocation(str_cur_allocate)
    port.yoc(str_yoc)
    port.yield_growth(str_yield_growth)
    port.update_portfolio_list(path_portfolio, str_cur_allocate)
    port.calculate_summary()
    # port.optimize(import_data=False)
    # port.optimize()
    port.export_csv("portfolio", m1_export_columns, str_cur_allocate)

    # print data analysis
    watch.print_terminal(export_columns, str_yield_ave, num_symbol=num_symbol)
    port.print_terminal(export_columns, str_yield_ave, num_symbol=num_symbol)
    port.print_summary(m1_export_columns, str_cur_allocate,
                       num_symbol=num_symbol)
    # port.print_terminal(export_columns, str_pe, ascending=True,
    #                     num_symbol=num_symbol)
    port.graph_history(path_history)
