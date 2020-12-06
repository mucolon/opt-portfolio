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


def performance_data(path, raw_columns):
    '''Convert performance data to a dataframe
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
    '''Convert dividend data to a dataframe
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


class Stocks:

    def __init__(self, dataframe, percent_columns, dollar_columns, round_columns):
        self.df = dataframe
        self.col_per = percent_columns
        self.col_dol = dollar_columns
        self.col_round = round_columns

    def yield_(self, sort=False):
        '''Add yield data column, will be FWD yield if present
            or TTM yield if FWD is missing
        '''
        str_ = "Yield"
        col_fwd_yield = self.df.columns.get_loc("Yield FWD")
        col_yields = [col_fwd_yield-1, col_fwd_yield]
        ave_yield = self.df.iloc[:, col_yields].mean(axis=1)
        self.fwd_yield = 2*ave_yield - self.df.iloc[:, col_fwd_yield-1]
        self.df.insert(col_fwd_yield+1, str_, self.fwd_yield)
        self.col_per = np.append(self.col_per, str_)

    def div_rate(self, sort=False):
        '''Add div rate data column, will be FWD div rate if present
            or TTM div rate if FWD is missing
        '''
        str_ = "Div Rate"
        col_fwd_rate = self.df.columns.get_loc("Div Rate FWD")
        col_rates = [col_fwd_rate-1, col_fwd_rate]
        ave_rate = self.df.iloc[:, col_rates].mean(axis=1)
        fwd_rate = 2*ave_rate - self.df.iloc[:, col_fwd_rate-1]
        self.df.insert(col_fwd_rate+1, str_, fwd_rate)
        self.col_dol = np.append(self.col_dol, str_)

    def ave_div_growth(self, sort=False):
        '''Add ave div growth data column
        '''
        str_ = "Ave Div Growth"
        col_5y_growth = self.df.columns.get_loc("Div Growth 5Y")
        col_growth = [col_5y_growth-1, col_5y_growth]
        self.ave_growth = self.df.iloc[:, col_growth].mean(axis=1)
        self.df.insert(col_5y_growth+1, str_, self.ave_growth)
        self.col_per = np.append(self.col_per, str_)

    def yoc(self, years, sort=False, sort_year=None):
        '''Add yoc data columns
        '''
        col_ave_grow = self.df.columns.get_loc("Ave Div Growth")
        _ = 1
        for i in years:
            yoc = self.fwd_yield*((1+self.ave_growth)**i)
            str_ = str(i) + "Y YoC"
            self.df.insert(col_ave_grow+_, str_, yoc)
            self.col_per = np.append(self.col_per, str_)
            _ += 1

    def filter_poor(self, year, exceptions=[]):
        '''Remove symbols based on poor yield, YoC, div growth, payout ratio, 3y, 5y, and 10y performances
        '''
        str_yoc = str(year) + "Y YoC"
        str_yield = "Yield"
        str_div_growth = "Ave Div Growth"
        str_3y_div_growth = "Div Growth 3Y"
        str_5y_div_growth = "Div Growth 5Y"
        str_payout = "Payout Ratio"
        str_3y_perf = "3Y Perf"
        str_3y_total = "3Y Total Return"
        str_5y_perf = "5Y Perf"
        str_5y_total = "5Y Total Return"
        str_10y_perf = "10Y Perf"
        str_10y_total = "10Y Total Return"
        str_years_growth = "Years of Growth"
        threshold_yoc = self.df.loc["SCHD", str_yoc]*1.1
        threshold_yield = self.df.loc["SCHD", str_yield]*1.1
        filt = (((self.df[str_yield] < threshold_yield) & (self.df[str_yoc] < threshold_yoc)) |
                (np.isnan(self.df[str_yield])) | (np.isnan(self.df[str_yoc])) |
                (np.isnan(self.df[str_3y_div_growth])) | (self.df[str_5y_div_growth] < 0) |
                (self.df[str_div_growth] < 0) | (self.df[str_div_growth] < 0) |
                (self.df[str_payout] > 0.95) | (self.df[str_3y_perf] < 0) | (self.df[str_3y_total] < 0) |
                (self.df[str_5y_perf] < 0) | (self.df[str_5y_total] < 0) |
                (self.df[str_10y_perf] < 0) | (self.df[str_10y_total] < 0) |
                (self.df[str_years_growth].str[0] == "0") | ((self.df[str_years_growth].str[0] == "1") &
                                                             (self.df[str_years_growth].str[1] == " ")))
        remove = self.df.loc[filt].index.values.tolist()
        if remove != []:
            [remove.remove(i) for i in exceptions if i in remove]
            self.df.drop(remove, inplace=True)
        self.remove_script = remove
        self.exceptions = exceptions

    def update_ignore_list(self, path_ignore):
        '''Update ignore list based on poor filter symbols found in filter_poor()
        '''
        ignore_df = pd.read_csv(path_ignore)
        col_index_market = ignore_df.columns.get_loc("Market")
        # index_portfolio = ignore_df["Portfolio"].dropna().tolist()
        cols_ignore = ignore_df.columns.values.tolist()
        col_script = cols_ignore[0]
        cols_remove = cols_ignore[1:col_index_market]
        _ = [ignore_df[i].dropna(
            how="all").values.tolist() for i in cols_remove]
        list_remove = [item for sublist in _ for item in sublist]
        self.df.drop(list_remove, inplace=True)
        if self.remove_script != []:
            script_df = pd.concat([pd.DataFrame([i], columns=[col_script])
                                   for i in self.remove_script], ignore_index=True)
            ignore_df.drop(columns=col_script, inplace=True)
            ignore_df = pd.concat([script_df, ignore_df], axis=1)
        ignore_df.to_csv(path_ignore, index=False)
        self.remove_total = list_remove

    def update_watchlist(self, path_list):
        '''Update watchlist based on poor filter symbols found in filter_poor()
        '''
        list_ = self.df.index.tolist()
        with open(path_list, "w") as f:
            f.write(','.join(list_))

        # with open(path_list, "r+") as f:
        #     raw_data = [line.strip("\n") for line in f]
        #     f.seek(0)
        #     f.write(','.join(raw_data))
        #     f.truncate()

    def sort(self, column, ascending=False):
        '''Sort dataframe specified column
        '''
        self.df.sort_values(column, inplace=True, ascending=ascending)

    def p2f_data(self):
        '''Convert string percent data columns to float number data columns
        '''
        for i in self.col_per:
            filt = (self.df.loc[:, i].str.contains("%"))
            edit = self.df.loc[filt, i].apply(p2f)
            self.df.loc[:, i] = edit

    def f2p_data(self):
        '''Convert float number data columns to string percent data columns
        '''
        for i in self.col_per:
            filt = (self.df.loc[:, i].apply(np.isnan) != True)
            edit = self.df.loc[filt, i].apply(f2p)
            self.df.loc[:, i] = edit

    def dollar2f_data(self):
        '''Convert string dollar data columns to float number data columns
        '''
        for i in self.col_dol:
            filt = (~self.df.loc[:, i].str.contains("-"))
            edit = self.df.loc[filt, i].apply(dollar2f)
            self.df.loc[:, i] = edit

    def f2dollar_data(self):
        '''Convert float number data columns to string percent data columns
        '''
        for i in self.col_dol:
            filt = (self.df.loc[:, i].apply(np.isnan) != True)
            edit = self.df.loc[filt, i].apply(f2dollar)
            self.df.loc[:, i] = edit

    def round_data(self, place=2):
        '''Round data columns to decimal place argument
        '''
        for i in self.col_round:
            filt = (self.df.loc[:, i].apply(np.isnan) != True)
            edit = self.df.loc[filt, i].apply(round, args=(place,))
            self.df.loc[:, i] = edit

    def export(self, file_name, columns):
        '''Export specificed dataframe columns to a input string file name
        '''
        cwd = os.getcwd()
        export_path = cwd + "/data/" + file_name + ".csv"
        self.df.loc[:, columns].to_csv(export_path)


if __name__ == "__main__":
    # pandas options
    pd.set_option("display.max_columns", None)
    # pd.set_option("display.max_rows", None)

    # file paths
    cwd = os.getcwd()
    path_ignore = cwd + "/data/ignore.csv"
    path_list = cwd + "/data/watchlist.txt"
    path_excel = cwd + "/data/Stocks.xlsx"

    # data constants
    percent_columns_perf = ["5D Perf", "1M Perf", "6M Perf", "YTD Perf", "1Y Perf", "3Y Perf",
                            "3Y Total Return", "5Y Perf", "5Y Total Return", "10Y Perf", "10Y Total Return"]
    percent_columns_div = ["Yield TTM", "Yield FWD", "4Y Avg Yield",
                           "Payout Ratio", "4Y Avg Payout", "Div Growth 3Y", "Div Growth 5Y"]
    dollar_columns_div = ["Div Rate TTM", "Div Rate FWD"]
    round_columns_value = ["P/E FWD"]
    percent_columns = percent_columns_perf + percent_columns_div
    dollar_columns = dollar_columns_div
    round_columns = round_columns_value

    # import data from excel file
    sheet_names = ["Performance", "Dividends", "Value"]
    df = excel_data(path_excel, sheet_names)
    stocks = Stocks(df, percent_columns, dollar_columns, round_columns)
    # start data analysis
    stocks.yield_()
    stocks.div_rate()
    stocks.ave_div_growth()
    years = [5, 10, 15, 20, 25]
    stocks.yoc(years)
    exceptions = ["DLR", "SCHD"]
    stocks.filter_poor(20, exceptions=exceptions)
    stocks.update_ignore_list(path_ignore)
    stocks.update_watchlist(path_list)
    watchlist_columns = ["Yield", "Ave Div Growth", "20Y YoC", "P/E FWD"]
    stocks.sort("P/E FWD", ascending=True)
    stocks.f2p_data()
    stocks.f2dollar_data()
    stocks.round_data()

    stocks.export("watchlist", watchlist_columns)
    _ = "="
    i = 19
    _ = str(i*_)
    _ = "\n" + _ + "\t" + _ + "\t" + _ + "\n"
    # print(_)
    print(stocks.df.loc[:, watchlist_columns])
    print(len(stocks.df.index))
