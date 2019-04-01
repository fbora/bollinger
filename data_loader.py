import os
import numpy as np
import pandas as pd
import re
import datetime
import logging
import itertools
import time

import config
import tws

TRADING_DAYS_IN_A_YEAR = 250


def get_data(start_date, end_date, training=False):
    data_file_template = config.loader.training_file_template if training else config.loader.inference_file_template
    today_str = datetime.datetime.today().strftime('%Y%m%d')
    if training: #fbora use a big cache we got for training
        today_str = '20181031'
    data_file = data_file_template.format(date=today_str)
    data_full_path = os.path.join(config.loader.data_dir, data_file)

    if os.path.isfile(data_full_path):
        with pd.HDFStore(data_full_path) as hdf:
            data = hdf['data']
    else:
        tickers = config.loader.tickers
        tws_hist_data = get_tws_hist_data(tickers, start_date, end_date)
        tws_today_data = get_tws_today_data(tickers)
        tws_full_data = pd.concat([tws_hist_data, tws_today_data], axis=0).reset_index(drop=True)

        bbg_data = create_bbg_fields(tws_full_data)
        open_data = get_open_features(bbg_data)

        window_feature_df_list = []
        window_period_params = list(itertools.product(config.loader.window_set, [max(config.loader.period_set)]))
        for w,p in window_period_params:
            etf_data_with_features = add_return_features(bbg_data, look_back_window=w, look_forward_window=w, periods=p)
            etf_data_with_features_adjusted = adjust_fwd_to_open(etf_data_with_features, open_data)
            etf_data_normalized = normalize_returns(etf_data_with_features_adjusted, bbg_data, scale_period=w)
            window_feature_df_list += [etf_data_with_features_adjusted, etf_data_normalized]

        data = pd.concat(window_feature_df_list, axis=1)

        # add PX to MA
        ma_features = [x for x in bbg_data.columns.unique(level=1) if 'MOV_AVG' in x] + ['PX_LAST']
        ma_multiidx = pd.MultiIndex.from_product([config.features.etf_tickers, ma_features])
        bbg_ma = bbg_data[ma_multiidx]
        moving_averages_normalized = normalize_moving_averages(bbg_ma)

        # add VIX data
        vix_columns = [x for x in bbg_data.columns.unique(level=1) if 'MOV_AVG' in x] + ['PX_LAST', 'CHG_PCT_1D']
        vix_multiidx = pd.MultiIndex.from_product([[config.features.vix_ticker], vix_columns])
        bbg_vix = bbg_data[vix_multiidx]
        vix_normalized = vix_features(bbg_vix)

        eod_data = pd.concat([data, moving_averages_normalized, vix_normalized], axis=1)

        # open data is computed relative to the open today.
        # eod data is computed relative to yesterday's close.
        # think pct change looking back zero days -> i can only compute percentage change after the close;
        # eod_data is one day behind open_data

        eod_data_shifted = eod_data.shift(1)
        data = pd.concat([eod_data_shifted, open_data], axis=1)

        data.sort_index(axis=1, inplace=True)
        data.sort_index(axis=0, ascending=True, inplace=True)
        data.to_hdf(data_full_path, 'data', format='t')


    feature_set = config.run.features_set + config.run.dependent_set
    null_count = pd.isnull(data.xs(config.run.dependent_set[0], level=1, axis=1)).sum()
    tickers = null_count[null_count.values < len(data.index)].index.values
    tickers = np.intersect1d(tickers, config.features.etf_tickers)
    data = data[pd.MultiIndex.from_product([tickers, feature_set])].copy()
    if training:
        data.dropna(inplace=True)

    return data[start_date:end_date]



def get_tws_hist_data(tickers, start_date, end_date, training=False):
    today_str = datetime.datetime.today().strftime('%Y%m%d')
    tws_file_name = config.loader.tws_hist_data_raw_file_template.format(date=today_str)
    tws_full_path = os.path.join(config.loader.data_dir, tws_file_name)

    # debug fbora change this here
    # tws_full_path = '/Users/florin/projects/sector_momentum/data/tws_20180625_raw_data.csv'
    if os.path.isfile(tws_full_path):
        with pd.HDFStore(tws_full_path) as hdf5_file:
            return hdf5_file['data']

    # convert start_date, end_date to duration in years
    start_datetime = datetime.datetime.strptime(re.sub("[^0-9]", "", start_date), '%Y%m%d')
    end_datetime = datetime.datetime.strptime(re.sub("[^0-9]", "", end_date), '%Y%m%d')
    time_diff = end_datetime - start_datetime

    end_date_tws = end_datetime.strftime('%Y%m%d 24:00:00')
    if time_diff.days > 365:
       duration_tws = '{} Y'.format(time_diff.days // 365 + 1)
    else:
        duration_tws = '{} d'.format(time_diff.days)

    app = tws.TWSapp()
    app.connect(config.loader.tws_host, config.loader.tws_port, clientId=123)
    if not app.isConnected():
        raise Exception("TWS not available for API connection.")
    print("serverVersion:%s connectionTime:%s" % (app.serverVersion(), app.twsConnectionTime()))

    # run the app in a separate thead
    thread_app = tws.ThreadedFunctor(app, 1, "Thread app")
    thread_app.start()

    ticker_df_list = []
    for tkr in tickers:
        logging.info('data_loader.get_tws_hist_data(): retrieving data for tkr={}'.format(tkr))
        resolved_contract = app.resolve_contract(tkr)
        tkr_df = app.get_historical_data(resolved_contract, end=end_date_tws, duration=duration_tws)
        tkr_df['ticker'] = tkr
        ticker_df_list.append(tkr_df)

    data = pd.concat(ticker_df_list, axis=0).reset_index(drop=True)
    data.to_hdf(tws_full_path, 'data', format='t')

    app.stop()
    thread_app.join()
    time.sleep(1)
    return data


def get_tws_today_data(tickers):
    today_str = datetime.datetime.today().strftime('%Y%m%d')
    tws_file_name = config.loader.tws_open_data_raw_file_template.format(date=today_str)
    tws_full_path = os.path.join(config.loader.data_dir, tws_file_name)

    if os.path.isfile(tws_full_path):
        with pd.HDFStore(tws_full_path) as hdf5_file:
            return hdf5_file['data']

    app = tws.TWSapp()
    app.connect(config.loader.tws_host, config.loader.tws_port, clientId=123)
    if not app.isConnected():
        raise Exception("TWS not available for API connection.")
    print("serverVersion:%s connectionTime:%s" % (app.serverVersion(), app.twsConnectionTime()))

    # run the app in a separate thead
    thread_app = tws.ThreadedFunctor(app, 1, "Thread app")
    thread_app.start()

    ticker_df_list = []
    for tkr in tickers:
        logging.info('data_loader.get_tws_today_data(): retrieving data for tkr={}'.format(tkr))
        resolved_contract = app.resolve_contract(tkr)
        tkr_df = app.get_open(resolved_contract)
        tkr_df['ticker'] = tkr
        ticker_df_list.append(tkr_df)

    data = pd.concat(ticker_df_list, axis=0).reset_index(drop=True)
    data.to_hdf(tws_full_path, 'data', format='t')

    app.stop()
    thread_app.join()
    return data


def create_bbg_fields(data):
    # create BBG FLDS - code is matches BBG convention to decimals and rounding
    data = data.sort_index(axis=1, ascending=True)
    data = data[['date', 'ticker', 'open', 'high', 'low', 'close']]
    data.date = pd.to_datetime(data.date, format='%Y%m%d')
    data['PX_LAST'] = data.close
    data['TOT_RETURN_INDEX_GROSS_DVD'] = data.close
    data_multi_index = pd.pivot_table(data, columns=['ticker'], index='date').swaplevel(axis=1)
    data_with_technicals = data_multi_index.sort_index(axis=1, ascending=True)
    data_with_technicals.columns.rename(['tickers', 'fields'], level=[0, 1], inplace=True)
    pct_change_values = np.around(data_with_technicals.xs('PX_LAST', level=1, axis=1).pct_change(1) * 100, decimals=4)
    # vix computed separately because BBG uses a different rounding convention:
    vix_pct_change = data_with_technicals['VIX'][['PX_LAST']].pct_change(1).round(4) * 100
    vix_pct_change.columns = ['CHG_PCT_1D']
    pct_change_values['VIX'] = vix_pct_change.values
    pct_change_values.columns = pd.MultiIndex.from_product([pct_change_values.columns, ['CHG_PCT_1D']])
    data_with_technicals[pct_change_values.columns] = pct_change_values

    data_with_technicals.sort_index(axis=1, inplace=True)
    chg_pct_zero_count = (data_with_technicals.xs('CHG_PCT_1D', level=1, axis=1)==0).sum(axis=1)
    tickers = data_with_technicals.columns.unique(level=0)
    if chg_pct_zero_count.max() == len(tickers):
        drop_dates = chg_pct_zero_count[chg_pct_zero_count == len(tickers)].index.values
        # drop with multiindex fails in pandas 0.23
        dates_to_keep = np.setdiff1d(data_with_technicals.index, drop_dates)
        data_with_technicals = data_with_technicals.loc[dates_to_keep]

    rounding_dict = dict.fromkeys(tickers, 4)
    rounding_dict['VIX'] = 2
    for ma in [5, 10, 30, 60, 100]:
        label_name = 'MOV_AVG_{}D'.format(ma)
        moving_average = data_with_technicals.xs('PX_LAST', level=1, axis=1).rolling(ma).mean().round(rounding_dict)
        moving_average.columns = pd.MultiIndex.from_product([moving_average.columns, [label_name]])
        data_with_technicals[moving_average.columns] = moving_average

    data_with_technicals.sort_index(ascending=True, inplace=True)

    rolling_one_day_std = data_with_technicals.xs('CHG_PCT_1D', level=1, axis=1).rolling(TRADING_DAYS_IN_A_YEAR).std() / 100.0
    rolling_one_day_std.columns = pd.MultiIndex.from_product([rolling_one_day_std.columns, ['STDEV_1D']])
    data_with_technicals[rolling_one_day_std.columns] = rolling_one_day_std

    data_with_technicals.sort_index(axis=1, inplace=True)
    return data_with_technicals


def add_return_features(bbg_data, look_back_window, look_forward_window, periods):
    tr_dvd = 'TOT_RETURN_INDEX_GROSS_DVD'
    rel_tr_windwow = 'TOTAL_RETURN_{}D_BK_{}D'.format(look_forward_window, 0)
    fwd_total_return = 'TOTAL_RETURN_{}D_FWD'.format(look_forward_window)
    px_name_template = 'TOTAL_RETURN_{}D_BK_{}D'

    data_cols = pd.MultiIndex.from_product([config.features.etf_tickers, ['TOT_RETURN_INDEX_GROSS_DVD']])
    tot_ret = bbg_data[data_cols]
    tot_ret = tot_ret.sort_index(axis=0, ascending=False)

    data = tot_ret.xs(tr_dvd, level=1, axis=1)/tot_ret.xs(tr_dvd, level=1, axis=1).shift(-look_back_window)-1.0
    data.columns = pd.MultiIndex.from_product([data.columns, [rel_tr_windwow]])

    fwd_total_return_df = tot_ret.xs(tr_dvd, level=1, axis=1).shift(look_back_window) / tot_ret.xs(tr_dvd, level=1, axis=1) - 1.0
    fwd_total_return_df.columns = pd.MultiIndex.from_product([fwd_total_return_df.columns, [fwd_total_return]])
    data = pd.concat([data, fwd_total_return_df], axis=1)

    data.sort_index(ascending=False, inplace=True)
    for i in range(1, periods):
        look_back_time = i * look_back_window
        px_name = px_name_template.format(look_back_window, look_back_time)
        px_shifted = data.xs(rel_tr_windwow, level=1, axis=1).shift(-look_back_time).copy()
        px_shifted.columns = pd.MultiIndex.from_product([px_shifted.columns, [px_name]])
        data = pd.concat([data, px_shifted], axis=1)

    data.sort_index(axis=1, inplace=True)
    return data


def normalize_returns(features, bbg_data, scale_period):
    # because rolling window can only look back you need to sort ascending
    features = features.sort_index(axis=0, ascending=True)
    data = None

    scaled_rolling_one_day_std = np.sqrt(scale_period) * bbg_data.xs('STDEV_1D', axis=1, level=1)
    return_cols = [x for x in features.columns.unique(level=1) if 'TOTAL_RETURN' in x]
    if return_cols:
        return_multi_idx = pd.MultiIndex.from_product([config.features.etf_tickers, return_cols])
        scaled_returns = features[return_multi_idx].divide(scaled_rolling_one_day_std, level=0, axis=1)
        scaled_column_map = dict(zip(return_cols, [x + '_SCALED' for x in return_cols]))
        scaled_returns.rename(columns=scaled_column_map, inplace=True)
        data = pd.concat([data, scaled_returns])

    data.sort_index(axis=1, inplace=True)
    data.sort_index(axis=0, ascending=False, inplace=True)
    return data


def normalize_moving_averages(data):
    # we want moving average relative to the last price
    ma_list = []
    ma_cols = [x for x in data.columns.unique(level=1) if 'MOV_AVG_' in x]
    if ma_cols:
        ma_cols_multiidx = pd.MultiIndex.from_product([data.columns.unique(level=0), ma_cols])
        ma_to_px = data[ma_cols_multiidx].divide(data.xs('PX_LAST', level=1, axis=1))
        px_to_ma = 1.0 / ma_to_px - 1.0
        ma_col_map = dict(zip(ma_cols, ['PX_LAST_TO_' + x for x in ma_cols]))
        px_to_ma.rename(columns=ma_col_map, inplace=True)
        ma_list.append(px_to_ma)
    ma_df = pd.concat(ma_list, axis=1)
    px_last = data.xs('PX_LAST', level=1, axis=1)
    px_last.columns = pd.MultiIndex.from_product([px_last.columns, ['PX_LAST']])
    ma_df = pd.concat([ma_df, px_last], axis=1)
    return ma_df


def vix_features(vix_bbg_data):
    vix_ma = normalize_moving_averages(vix_bbg_data)
    px_chg = vix_bbg_data[pd.MultiIndex.from_product([[config.features.vix_ticker], ['CHG_PCT_1D']])]
    vix_norm = pd.concat([vix_ma, px_chg], axis=1)
    ks = vix_norm.columns.unique(level=1)
    vix_dict = dict(zip(ks, ['VIX_' + x for x in ks]))
    vix_norm.rename(columns=vix_dict, inplace=True)

    data = None
    for tkr in config.features.etf_tickers:
        tkr_vix_predictors = vix_norm.copy()
        tkr_vix_predictors.columns = pd.MultiIndex.from_product([[tkr], vix_norm.columns.unique(level=1)])
        data = pd.concat([data, tkr_vix_predictors], axis=1)

    return data


def get_open_features(bbg_data):
    open_to_yday_close_relative = bbg_data.xs('open', level=1, axis=1) / \
        bbg_data.xs('close', level=1, axis=1).shift(1) - 1.0
    open_to_yday_close_relative.columns = pd.MultiIndex.from_product(
        [open_to_yday_close_relative.columns, ['OPEN_TO_YDAY_CLOSE_RETURN']])

    close_to_open_relative = bbg_data.xs('close', level=1, axis=1) / \
        bbg_data.xs('open', level=1, axis=1) - 1.0
    close_to_open_relative.columns = pd.MultiIndex.from_product(
        [close_to_open_relative.columns, ['CLOSE_TO_OPEN_RETURN']])

    # scale variables
    stdev = bbg_data.xs('STDEV_1D', axis=1, level=1)
    open_price = bbg_data.xs('open', axis=1, level=1)
    open_to_yday_close_relative_scaled = open_to_yday_close_relative.divide(stdev, level=0, axis=1)
    open_to_yday_close_relative_scaled.rename(columns={'OPEN_TO_YDAY_CLOSE_RETURN':'OPEN_TO_YDAY_CLOSE_RETURN_SCALED'}, inplace=True)
    close_to_open_relative_scaled = close_to_open_relative.divide(stdev, level=0, axis=1)
    close_to_open_relative_scaled.rename(columns={'CLOSE_TO_OPEN_RETURN':'CLOSE_TO_OPEN_RETURN_SCALED'}, inplace=True)
    stdev.columns = pd.MultiIndex.from_product([stdev.columns, ['STDEV_1D']])
    open_price.columns = pd.MultiIndex.from_product([open_price.columns, ['OPEN']])

    all_open = pd.concat([open_to_yday_close_relative, open_to_yday_close_relative_scaled,
        close_to_open_relative, close_to_open_relative_scaled, stdev, open_price], axis=1)

    etf_open = all_open[config.features.etf_tickers]

    vix_open = all_open[config.features.vix_ticker].drop('OPEN', axis=1)
    # vix_open = all_open[config.features.vix_ticker]
    vix_open_df = bbg_data[config.features.vix_ticker][['open']]/100.0
    vix_open_df.columns=['OPEN']
    vix_open = pd.concat([vix_open, vix_open_df], axis=1)
    vix_open.columns = ['VIX_' + x for x in vix_open.columns]

    for tkr in config.features.etf_tickers:
        tkr_vix_open = vix_open.copy()
        tkr_vix_open.columns = pd.MultiIndex.from_product([[tkr], tkr_vix_open.columns])
        etf_open = pd.concat([etf_open, tkr_vix_open], axis=1)

    etf_open.sort_index(axis=1, inplace=True)

    return etf_open


def adjust_fwd_to_open(etf_data_with_features, open_data):
    etf_data_with_features_adjusted = etf_data_with_features.copy()
    fwd_scaled_columns = [x for x in etf_data_with_features.columns.unique(level=1) if '_FWD' in x]
    fwd_multi = pd.MultiIndex.from_product([etf_data_with_features.columns.unique(level=0), fwd_scaled_columns])
    fwd_df = etf_data_with_features[fwd_multi].copy()
    open_to_yday_close = open_data.xs('OPEN_TO_YDAY_CLOSE_RETURN', level=1, axis=1)
    open_to_yday_close_shifted = open_to_yday_close.shift(-1)
    fwd_df_adjusted = fwd_df.subtract(open_to_yday_close_shifted, level=0, axis=1)
    etf_data_with_features_adjusted[fwd_df_adjusted.columns] = fwd_df_adjusted
    return etf_data_with_features_adjusted


def split_x_y(data):
    x = data[config.run.features_set]
    y = data[config.run.dependent_set]
    return x, y


def get_vix_open():
    # assumes that today's file was saved already:
    tws_today_data = get_tws_today_data(config.loader.tickers)
    vix_open = tws_today_data[tws_today_data.ticker == 'VIX'].open.values[0]
    return vix_open


def get_etf_account_positions():
    app = tws.TWSapp()
    app.connect(config.loader.tws_host, config.loader.tws_port, clientId=123)
    if not app.isConnected():
        raise Exception("TWS not available for API connection.")
    print("serverVersion:%s connectionTime:%s" % (app.serverVersion(), app.twsConnectionTime()))

    # run the app in a separate thead
    thread_app = tws.ThreadedFunctor(app, 1, "Thread app")
    thread_app.start()

    positions = app.get_account_positions()

    etf_positions = positions[(positions.sec_type == 'STK') &
        positions.ticker.isin(config.trade_params.ticker_params.keys()) &
        (positions.shares != 0)]

    app.stop()
    thread_app.join()
    time.sleep(1)

    return etf_positions


def main():
    pos = get_etf_account_positions()
    print(pos)
    return
    # get_vix_open()
    start_date = '20170101'
    end_date = datetime.datetime.today().strftime('%Y%m%d')

    data = get_data(start_date, end_date, False)
    data.head()


if __name__ == '__main__':
    main()
