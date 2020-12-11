# Setup code
import numpy as np
import pandas as pd
import os


def p2f(percent):
    '''Convert string percent to a float number
    '''
    return float(percent.strip("%"))/100


def f2p(float_):
    '''Convert float number to a string percent
    '''
    return str(round(100*float_, 2))+"%"


def dollar2f(dollar):
    '''Convert string dollar to float number
    '''
    return float(dollar.strip("$"))


def f2dollar(float_):
    '''Convert float number to string dollar
    '''
    dollar = "$" + str(round(float_, 2))
    length = len(dollar)
    decimal = dollar.index(".")
    if (length - decimal) == 2:
        dollar = dollar + "0"
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
    raw_rows = int(len_data/raw_columns)
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
    raw_rows = int(len_data/raw_columns)
    entries = raw_rows - 1
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
    df.replace("-", np.nan, inplace=True)
    return df


class Watchlist:
    '''Class filters Stocks.xlsx to a watchlist dataframe that passes the filter_poor() method
    '''

    def __init__(self, dataframe, percent_columns, dollar_columns, round_columns):
        self.df = dataframe
        self.col_per = percent_columns
        self.col_dol = dollar_columns
        self.col_round = round_columns

    def yield_(self, str_):
        '''Add yield data column, will be FWD yield if present or TTM yield if FWD is missing
        '''
        self.str_yield = str_
        col_fwd_yield = self.df.columns.get_loc("Yield FWD")
        col_yields = [col_fwd_yield-1, col_fwd_yield]
        ave_yield = self.df.iloc[:, col_yields].mean(axis=1)
        self.fwd_yield = 2*ave_yield - self.df.iloc[:, col_fwd_yield-1]
        self.df.insert(col_fwd_yield+1, str_, self.fwd_yield)
        self.col_per = np.append(self.col_per, str_)

    def div_rate(self, str_):
        '''Add div rate data column, will be FWD div rate if present or TTM div rate if FWD is missing
        '''
        self.str_div_rate = str_
        col_fwd_rate = self.df.columns.get_loc("Div Rate FWD")
        col_rates = [col_fwd_rate-1, col_fwd_rate]
        ave_rate = self.df.iloc[:, col_rates].mean(axis=1)
        fwd_rate = 2*ave_rate - self.df.iloc[:, col_fwd_rate-1]
        self.df.insert(col_fwd_rate+1, str_, fwd_rate)
        self.col_dol = np.append(self.col_dol, str_)

    def ave_div_growth(self, str_):
        '''Add ave div growth data column
        '''
        self.str_div_growth = str_
        col_5y_growth = self.df.columns.get_loc("Div Growth 5Y")
        col_growth = [col_5y_growth-1, col_5y_growth]
        self.ave_growth = self.df.iloc[:, col_growth].mean(axis=1)
        self.df.insert(col_5y_growth+1, str_, self.ave_growth)
        self.col_per = np.append(self.col_per, str_)

    def yoc_years(self, years):
        '''Add projected yoc data columns
        '''
        col_ave_grow = self.df.columns.get_loc(self.str_div_growth)
        _ = 1
        for i in years:
            yoc = self.fwd_yield*((1+self.ave_growth)**i)
            str_ = str(i) + "Y YoC"
            self.df.insert(col_ave_grow+_, str_, yoc)
            self.col_per = np.append(self.col_per, str_)
            _ += 1

    def pe_ratio(self, str_):
        '''Add P/E ratio data column, will be FWD P/E if present or TTM P/E if FWD P/E is missing
        '''
        self.str_pe = str_
        col_fwd_pe = self.df.columns.get_loc("P/E FWD")
        col_pes = [col_fwd_pe-1, col_fwd_pe]
        ave_pe = self.df.iloc[:, col_pes].mean(axis=1)
        filt = (ave_pe == self.df.iloc[:, col_fwd_pe])
        fwd_pe = 2*ave_pe - self.df.iloc[:, col_fwd_pe-1]
        fwd_pe[filt] = ave_pe
        self.df.insert(col_fwd_pe+1, str_, fwd_pe)
        self.col_round = np.append(self.col_round, str_)

    def filter_poor(self, str_, exceptions=None):
        '''Remove symbols based on poor yield, YoC, div growth, payout ratio, 3y, 5y, and 10y performances
        '''
        if exceptions is None:
            exceptions = []
        str_yoc = str_
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
        threshold_yoc = self.df.loc["SCHD", str_yoc]*1.1
        threshold_yield = self.df.loc["SCHD", str_yield]*1.1
        filt = (((self.df[str_yield] < threshold_yield) & (self.df[str_yoc] < threshold_yoc)) |
                ((self.df[str_years_growth].str[0] == "1") & (self.df[str_years_growth].str[1] == " ")) |
                (self.df[str_years_growth].str[0] == "0") | (self.df[str_payout] > 0.95) |
                (np.isnan(self.df[str_yield])) | (np.isnan(self.df[str_yoc])) |
                (np.isnan(self.df[str_3y_div_growth])) | (self.df[str_5y_div_growth] < 0) |
                (self.df[str_div_growth] < 0) | (self.df[str_div_growth] < 0) |
                (self.df[str_3y_perf] < 0) | (self.df[str_3y_total] < 0) |
                (self.df[str_5y_perf] < 0) | (self.df[str_5y_total] < 0) |
                (self.df[str_10y_perf] < 0) | (self.df[str_10y_total] < 0))
        remove_script = self.df.loc[filt].index.values.tolist()
        self.remove_script = remove_script
        self.exceptions = exceptions

    def update_ignore_list(self, path_ignore):
        '''Update ignore csv based on symbols found after filter_poor() method
        '''
        remove_script = self.remove_script
        exceptions = self.exceptions
        ignore_df = pd.read_csv(path_ignore)
        col_index_market = ignore_df.columns.get_loc("Market")
        index_portfolio = ignore_df["Portfolio"].dropna().tolist()
        index_market = ignore_df["Market"].dropna().tolist()
        cols_ignore = ignore_df.columns.values.tolist()
        col_script = cols_ignore[0]
        cols_qual = cols_ignore[1:col_index_market]
        _ = [ignore_df[i].dropna(
            how="all").values.tolist() for i in cols_qual]
        remove_qual = [item for sublist in _ for item in sublist]
        self.remove_qual = remove_qual
        entries = self.df.index.tolist()
        remove_qual = [i for i in remove_qual if i in entries]
        remove_qual = [i for i in remove_qual if i not in exceptions]
        self.df.drop(remove_qual, inplace=True)
        entries = self.df.index.tolist()
        exceptions = exceptions + index_market + index_portfolio
        exceptions = remove_duplicates(exceptions)
        if remove_script != []:
            remove_script = [i for i in remove_script if i not in exceptions]
            remove_script = [i for i in remove_script if i in entries]
            self.df.drop(remove_script, inplace=True)
            script_df = pd.concat([pd.DataFrame([i], columns=[col_script])
                                   for i in remove_script], ignore_index=True)
            ignore_df.drop(columns=col_script, inplace=True)
            ignore_df = pd.concat([script_df, ignore_df], axis=1)
        ignore_df.to_csv(path_ignore, index=False)
        self.index_market = index_market
        self.index_portfolio = index_portfolio

    def update_watchlist(self, path_list):
        '''Update watchlist text file based on symbols that passed filter_poor() method
        '''
        entries = self.df.index.tolist()
        entries = [i for i in entries if i not in self.remove_qual]
        with open(path_list, "w") as f:
            f.write(','.join(entries))

    def sort(self, column, ascending=False):
        '''Sort dataframe based on specified column
        '''
        self.df.sort_values(column, inplace=True, ascending=ascending)

    def allocate(self, column, str_):
        '''Allocate portfolio distribution based on input column
        '''
        self.str_tar_allocate = str_
        index_portfolio = self.index_portfolio
        sum_ = self.df.loc[index_portfolio, column].sum()
        self.df[str_] = (1/sum_)*self.df.loc[index_portfolio, column]
        round_ = self.df.loc[index_portfolio, str_].apply(round, args=(2,))
        self.df.loc[index_portfolio, str_] = round_
        self.col_per = np.append(self.col_per, str_)

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
                filt = (new_df.loc[:, i].apply(np.isnan) != True)
                edit = new_df.loc[filt, i].apply(f2p)
                new_df.loc[:, i] = edit
            return new_df
        else:
            for i in self.col_per:
                filt = (self.df.loc[:, i].apply(np.isnan) != True)
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
                filt = (new_df.loc[:, i].apply(np.isnan) != True)
                edit = new_df.loc[filt, i].apply(f2dollar)
                new_df.loc[:, i] = edit
            return new_df
        else:
            for i in self.col_dol:
                filt = (self.df.loc[:, i].apply(np.isnan) != True)
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
                filt = (new_df.loc[:, i].apply(np.isnan) != True)
                edit = new_df.loc[filt, i].apply(round, args=(place,))
                new_df.loc[:, i] = edit
            return new_df
        else:
            for i in self.col_round:
                filt = (self.df.loc[:, i].apply(np.isnan) != True)
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

    def print_terminal(self, columns, sort_column, ascending=False):
        '''Print specificed dataframe columns onto terminal
        '''
        _ = "="
        i = 16
        _ = str(i*_)
        _ = _ + "\t" + _ + "\t" + _
        print(_)

        self.sort(sort_column, ascending=ascending)
        df = self.cleanup_data(return_dataframe=True)
        print(df.loc[:, columns])
        print(len(df.index))

    def export_csv(self, file_name, columns, sort_column, ascending=False):
        '''Export specificed dataframe columns to a csv
        '''
        self.sort(sort_column, ascending=ascending)
        df = self.cleanup_data(return_dataframe=True)
        cwd = os.getcwd()
        export_path = cwd + "/data/" + file_name + ".csv"
        df.loc[:, columns].to_csv(export_path)


class Portfolio(Watchlist):
    '''Class highlights portfolio performance
    '''

    def __init__(self, watchlist_obj):
        watch = watchlist_obj
        self.index_market = watch.index_market
        self.exceptions = watch.exceptions
        self.index_portfolio = watch.index_portfolio
        self.str_yield = watch.str_yield
        self.str_div_rate = watch.str_div_rate
        index_ = watch.index_portfolio + watch.exceptions + watch.index_market
        index_ = remove_duplicates(index_)
        self.df = watch.df.loc[index_, :]
        self.col_per = watch.col_per
        self.col_dol = watch.col_dol
        self.col_round = watch.col_round

    def update_portfolio_list(self, path_list):
        '''Update portfolio list based on ignore csv portfolio column
        '''
        list_ = self.index_portfolio + self.index_market
        with open(path_list, "w") as f:
            f.write(','.join(list_))

    def m1_import(self, path_csv, percent_columns, dollar_columns):
        '''Import M1 csv file for portfolio analysis
        '''
        m1 = pd.read_csv(path_csv)
        m1.set_index("Ticker", inplace=True)
        m1.rename_axis("Symbols", inplace=True)
        m1.rename(columns={"Avg. Price": "Ave Price"}, inplace=True)
        self.df = pd.concat([self.df, m1], axis=1)
        self.col_per = np.append(self.col_per, percent_columns)
        self.col_dol = np.append(self.col_dol, dollar_columns)
        self.p2f_data()
        self.dollar2f_data()

    def annual_div(self, str_):
        '''Add portfolio annual div rate data column based on shares, calculate total portfolio div rate based on shares
        '''
        self.str_annual_div = str_
        div_annual = self.df[self.str_div_rate]*self.df["Shares"]
        self.df[str_] = div_annual
        self.portfolio_annual_div = self.df[str_].sum()
        self.col_dol = np.append(self.col_dol, str_)

    def current_allocation(self, str_):
        '''Add portfolio current allocation data amount based on market value of shares
        '''
        self.str_cur_allocate = str_
        str_value = "Value"
        total_invest = self.df[str_value].sum()
        allocation = self.df[str_value]/total_invest
        self.df[str_] = allocation
        self.col_per = np.append(self.col_per, str_)

    def yoc(self, str_):
        '''Add portfolio YoC data column based on cost basis
        '''
        self.str_yoc = str_
        yoc = self.df[self.str_div_rate]/self.df["Ave Price"]
        self.df[str_] = yoc
        self.col_per = np.append(self.col_per, str_)

    def yield_growth(self, str_):
        '''Add portfolio div growth data column
        '''
        self.str_yield_growth = str_
        yield_growth = (self.df[self.str_yoc]/self.df[self.str_yield]) - 1
        self.df[str_] = yield_growth
        self.col_per = np.append(self.col_per, str_)

    def averages(self):
        '''Calculate portfolio average yield, yoc, and yield growth
        '''
        self.ave_yield = sum(self.df[self.str_cur_allocate] *
                             self.df[self.str_yield])
        self.ave_yoc = sum(self.df[self.str_cur_allocate] *
                           self.df[self.str_yoc])
        self.ave_yield_growth = sum(self.df[self.str_cur_allocate] *
                                    self.df[self.str_yield_growth])

    def print_summary(self, columns, sort_column, ascending=False):
        '''Print specificed dataframe columns and portfolio summary onto terminal
        '''
        self.portfolio_annual_div = f2dollar(self.portfolio_annual_div)
        self.ave_yield = f2p(self.ave_yield)
        self.ave_yoc = f2p(self.ave_yoc)
        self.ave_yield_growth = f2p(self.ave_yield_growth)

        _ = "="
        i = 16
        _ = str(i*_)
        _ = _ + "\t" + _ + "\t" + _
        print(_)

        self.sort(sort_column, ascending=ascending)
        df = self.cleanup_data(return_dataframe=True)
        print(df.loc[:, columns])
        print(len(df.index))
        print("Annual Portfolio Dividends: ", self.portfolio_annual_div)
        print("Average Portfolio Yield: ", self.ave_yield)
        print("Average Portfolio YoC: ", self.ave_yoc)
        print("Average Portfolio Yield Growth: ", self.ave_yield_growth)


if __name__ == "__main__":
    # pandas options
    # pd.set_option("display.max_columns", None)
    # pd.set_option("display.max_rows", None)

    # file paths
    cwd = os.getcwd()
    path_ignore = cwd + "/data/ignore.csv"
    path_list = cwd + "/data/watchlist.txt"
    path_portfolio = cwd + "/data/portfolio.txt"
    path_excel = cwd + "/data/Stocks.xlsx"
    path_m1 = cwd + "/personal/m1.csv"

    # data constants
    percent_columns_perf = ["5D Perf", "1M Perf", "6M Perf", "YTD Perf", "1Y Perf", "3Y Perf",
                            "3Y Total Return", "5Y Perf", "5Y Total Return", "10Y Perf", "10Y Total Return"]
    percent_columns_div = ["Yield TTM", "Yield FWD", "4Y Avg Yield",
                           "Payout Ratio", "4Y Avg Payout", "Div Growth 3Y", "Div Growth 5Y"]
    percent_columns_m1 = ["Unrealized Gain %"]
    dollar_columns_div = ["Div Rate TTM", "Div Rate FWD"]
    dollar_columns_m1 = ["Ave Price", "Cost Basis", "Unrealized Gain", "Value"]
    round_columns_value = ["P/E TTM", "P/E FWD"]
    percent_columns = percent_columns_perf + percent_columns_div
    dollar_columns = dollar_columns_div
    round_columns = round_columns_value

    # import data from excel file
    sheet_names = ["Performance", "Dividends", "Value"]
    df = excel_data(path_excel, sheet_names)

    # data analysis constants
    str_yield = "Yield"
    str_div_rate = "Div Rate"
    str_div_growth = "Ave Div Growth"
    str_pe = "P/E"
    str_tar_allocate = "Target Allocation"
    exceptions = ["HRL", "LOW", "ITW", "BKH"]
    years = [5, 10, 15, 20]
    yoc_year = years[-1]
    str_yoc_year = str(yoc_year) + "Y YoC"
    export_columns = [str_yield, str_div_growth,
                      str_yoc_year, str_pe, str_tar_allocate]
    str_annual_div = "Annual Div"
    str_cur_allocate = "Current Allocation"
    str_yoc = "YoC"
    str_yield_growth = "Div Growth"
    str_value = "Value"
    m1_export_columns = [str_tar_allocate, str_cur_allocate, "Shares", "Ave Price", "Cost Basis",
                         str_value, "Unrealized Gain %", "Annual Div", str_yield, str_yoc, str_yield_growth]

    # start data analysis to filter stocks to a singular watchlist
    watch = Watchlist(df, percent_columns, dollar_columns, round_columns)
    watch.yield_(str_yield)
    watch.div_rate(str_div_rate)
    watch.ave_div_growth(str_div_growth)
    watch.yoc_years(years)
    watch.pe_ratio(str_pe)
    watch.filter_poor(str_yoc_year, exceptions=exceptions)
    watch.sort(str_yield)
    watch.update_ignore_list(path_ignore)
    watch.update_watchlist(path_list)
    watch.allocate(str_yield, str_tar_allocate)
    watch.export_csv("watchlist", export_columns, str_yield)

    # start data analysis to highlight portfolio performance
    port = Portfolio(watch)
    port.update_portfolio_list(path_portfolio)
    port.m1_import(path_m1, percent_columns_m1, dollar_columns_m1)
    port.annual_div(str_annual_div)
    port.current_allocation(str_cur_allocate)
    port.yoc(str_yoc)
    port.yield_growth(str_yield_growth)
    port.averages()
    port.export_csv("portfolio", m1_export_columns, str_yield)

    # print data
    # print(port.df.info())
    # watch.print_terminal(export_columns, str_yield)
    # port.print_terminal(export_columns, str_yield)
    # port.print_terminal(export_columns, str_pe, ascending=True)
    port.print_summary(m1_export_columns, str_value)
