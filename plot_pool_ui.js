function sma(arr, n) {
      const out = new Array(arr.length).fill(null);
      if (arr.length < n) return out;
      let s = 0;
      for (let i = 0; i < n; i++) s += arr[i];
      out[n - 1] = +(s / n).toFixed(4);
      for (let i = n; i < arr.length; i++) {
        s += arr[i] - arr[i - n];
        out[i] = +(s / n).toFixed(4);
      }
      return out;
    }

    /** EMA；跳过前导 null，用前 period 个有效值做 SMA 种子。 */
    function ema(values, period) {
      const out = new Array(values.length).fill(null);
      const k = 2 / (period + 1);
      let prev = null;
      const buf = [];
      for (let i = 0; i < values.length; i++) {
        const v = values[i];
        if (v == null || Number.isNaN(v)) continue;
        if (prev === null) {
          buf.push(v);
          if (buf.length === period) {
            prev = buf.reduce((a, b) => a + b, 0) / period;
            out[i] = +prev.toFixed(4);
          }
        } else {
          prev = v * k + prev * (1 - k);
          out[i] = +prev.toFixed(4);
        }
      }
      return out;
    }

    /** 国内常用 MACD(12,26,9)：DIF / DEA / MACD柱=(DIF-DEA)*2 */
    function calcMacd(closes, fast, slow, signal) {
      fast = fast || 12;
      slow = slow || 26;
      signal = signal || 9;
      const emaFast = ema(closes, fast);
      const emaSlow = ema(closes, slow);
      const dif = closes.map((_, i) => {
        if (emaFast[i] == null || emaSlow[i] == null) return null;
        return +(emaFast[i] - emaSlow[i]).toFixed(4);
      });
      const dea = ema(dif, signal);
      const macd = dif.map((d, i) => {
        if (d == null || dea[i] == null) return null;
        return +((d - dea[i]) * 2).toFixed(4);
      });
      return { dif, dea, macd };
    }

    /** 短线看近端均线，长线看中期；另一条默认隐藏，图例可点开。 */
    function maLegendSelected(panel) {
      const isShort = panel === 'short' || panel === 'scalp';
      return {
        MA5: isShort,
        MA10: true,
        MA20: true,
        MA60: !isShort
      };
    }

    function boxMarkLine(stock) {
      const lines = [];
      if (stock.boxBottom != null && Number.isFinite(stock.boxBottom)) {
        lines.push({
          yAxis: stock.boxBottom,
          name: '箱体底',
          label: {
            formatter: '箱体底 ' + Number(stock.boxBottom).toFixed(2),
            color: '#c62828',
            fontSize: 10,
            position: 'insideEndTop'
          },
          lineStyle: { type: 'dashed', color: '#c62828', width: 1.4 }
        });
      }
      if (stock.takeProfit != null && Number.isFinite(stock.takeProfit)) {
        lines.push({
          yAxis: stock.takeProfit,
          name: '止盈',
          label: {
            formatter: '止盈 ' + Number(stock.takeProfit).toFixed(2),
            color: '#0d7a5f',
            fontSize: 10,
            position: 'insideEndBottom'
          },
          lineStyle: { type: 'dashed', color: '#0d7a5f', width: 1.1 }
        });
      }
      if (!lines.length) return undefined;
      return {
        silent: true,
        symbol: 'none',
        data: lines
      };
    }

    function buildOption(stock, panel) {
      const bars = stock.bars;
      const dates = bars.map(b => b.date);
      const ohlc = bars.map(b => [b.open, b.close, b.low, b.high]);
      const amounts = bars.map(b => b.amountYi);
      const volumes = bars.map(b => +(b.volume / 1e8).toFixed(4));
      const closes = bars.map(b => b.close);
      const ma5 = sma(closes, 5);
      const ma10 = sma(closes, 10);
      const ma20 = sma(closes, 20);
      const ma60 = sma(closes, 60);
      const m = calcMacd(closes, 12, 26, 9);
      // 与 K 线一致：涨红跌绿
      const amountColors = bars.map(b => b.close >= b.open ? '#e74c3c' : '#14b15b');
      const volumeColors = bars.map(b => b.close >= b.open ? '#e74c3c' : '#14b15b');
      const macdColors = m.macd.map(v => (v == null || v >= 0) ? '#e74c3c' : '#14b15b');

      const amountData = amounts.map((v, i) => ({
        value: v,
        itemStyle: { color: amountColors[i] }
      }));
      const volumeData = volumes.map((v, i) => ({
        value: v,
        itemStyle: { color: volumeColors[i] }
      }));
      const macdData = m.macd.map((v, i) => ({
        value: v,
        itemStyle: { color: macdColors[i] }
      }));
      const tab = panel || (typeof currentTab !== 'undefined' ? currentTab : 'long');

      return {
        animation: false,
        legend: {
          data: ['K线', 'MA5', 'MA10', 'MA20', 'MA60', '成交额', '成交量', 'DIF', 'DEA', 'MACD'],
          selected: maLegendSelected(tab),
          top: 0,
          textStyle: { fontSize: 11 }
        },
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'cross' },
          borderWidth: 1,
          borderColor: '#ccc',
          textStyle: { fontSize: 12 },
          formatter: function (params) {
            if (!params || !params.length) return '';
            const i = params[0].dataIndex;
            const b = bars[i];
            const prev = i > 0 ? bars[i - 1].close : null;
            const chg = prev && prev !== 0 ? ((b.close - prev) / prev * 100) : null;
            const chgTxt = chg == null ? '—' : ((chg >= 0 ? '+' : '') + chg.toFixed(2));
            let html = `<b>${b.date}</b><br/>开 ${b.open} 高 ${b.high}<br/>低 ${b.low} 收 ${b.close} (${chgTxt}%)` +
              `<br/>成交额 ${b.amountYi.toFixed(4)}亿` +
              `<br/>成交量 ${b.volume.toLocaleString()}股 (${(b.volume / 1e8).toFixed(4)}亿)`;
            if (m.dif[i] != null) html += `<br/>DIF ${m.dif[i]}`;
            if (m.dea[i] != null) html += `<br/>DEA ${m.dea[i]}`;
            if (m.macd[i] != null) html += `<br/>MACD ${m.macd[i]}`;
            for (const p of params) {
              // 注意：不能用 startsWith('MA')，会误伤 MACD
              if ((p.seriesName === 'MA5' || p.seriesName === 'MA10'
                  || p.seriesName === 'MA20' || p.seriesName === 'MA60')
                  && p.data != null && typeof p.data !== 'object')
                html += `<br/>${p.marker}${p.seriesName} ${p.data}`;
            }
            return html;
          }
        },
        axisPointer: { link: [{ xAxisIndex: 'all' }] },
        grid: [
          { left: 52, right: 18, top: 40, height: '38%' },
          { left: 52, right: 18, top: '50%', height: '10%' },
          { left: 52, right: 18, top: '62%', height: '10%' },
          { left: 52, right: 18, top: '74%', height: '14%' }
        ],
        xAxis: [
          {
            type: 'category', data: dates, boundaryGap: true,
            axisLine: { onZero: false }, splitLine: { show: false },
            min: 'dataMin', max: 'dataMax', axisLabel: { show: false }
          },
          {
            type: 'category', gridIndex: 1, data: dates, boundaryGap: true,
            axisLine: { onZero: false }, axisTick: { show: false },
            splitLine: { show: false }, min: 'dataMin', max: 'dataMax',
            axisLabel: { show: false }
          },
          {
            type: 'category', gridIndex: 2, data: dates, boundaryGap: true,
            axisLine: { onZero: false }, axisTick: { show: false },
            splitLine: { show: false }, min: 'dataMin', max: 'dataMax',
            axisLabel: { show: false }
          },
          {
            type: 'category', gridIndex: 3, data: dates, boundaryGap: true,
            axisLine: { onZero: false }, axisTick: { show: false },
            splitLine: { show: false }, min: 'dataMin', max: 'dataMax',
            axisLabel: { fontSize: 10, hideOverlap: true }
          }
        ],
        yAxis: [
          {
            scale: true, splitArea: { show: true },
            axisLabel: { fontSize: 10 }
          },
          {
            scale: true, gridIndex: 1, splitNumber: 2,
            axisLabel: { show: false }, axisLine: { show: false },
            axisTick: { show: false }, splitLine: { show: false }
          },
          {
            scale: true, gridIndex: 2, splitNumber: 2,
            axisLabel: { show: false }, axisLine: { show: false },
            axisTick: { show: false }, splitLine: { show: false }
          },
          {
            scale: true, gridIndex: 3, splitNumber: 3,
            axisLabel: { fontSize: 9 }, axisLine: { show: false },
            axisTick: { show: false }, splitLine: { show: true, lineStyle: { type: 'dashed', opacity: 0.4 } }
          }
        ],
        dataZoom: [
          { type: 'inside', xAxisIndex: [0, 1, 2, 3], start: 0, end: 100 },
          {
            show: true, xAxisIndex: [0, 1, 2, 3], type: 'slider',
            top: '92%', height: 16, start: 0, end: 100,
            borderColor: 'transparent'
          }
        ],
        series: [
          {
            name: 'K线',
            type: 'candlestick',
            data: ohlc,
            itemStyle: {
              color: '#e74c3c',
              color0: '#14b15b',
              borderColor: '#e74c3c',
              borderColor0: '#14b15b'
            },
            markLine: boxMarkLine(stock)
          },
          {
            name: 'MA5', type: 'line', data: ma5,
            smooth: false, showSymbol: false,
            lineStyle: { width: 1.2, color: '#16a085' },
            itemStyle: { color: '#16a085' }
          },
          {
            name: 'MA10', type: 'line', data: ma10,
            smooth: false, showSymbol: false,
            lineStyle: { width: 1.2, color: '#f39c12' },
            itemStyle: { color: '#f39c12' }
          },
          {
            name: 'MA20', type: 'line', data: ma20,
            smooth: false, showSymbol: false,
            lineStyle: { width: 1.2, color: '#3498db' },
            itemStyle: { color: '#3498db' }
          },
          {
            name: 'MA60', type: 'line', data: ma60,
            smooth: false, showSymbol: false,
            lineStyle: { width: 1.2, color: '#9b59b6' },
            itemStyle: { color: '#9b59b6' }
          },
          {
            name: '成交额',
            type: 'bar',
            xAxisIndex: 1,
            yAxisIndex: 1,
            data: amountData,
            // 固定色给图例用（否则会跟 K 线一样变成默认红）
            itemStyle: { color: '#5b8ff9' },
            color: '#5b8ff9'
          },
          {
            name: '成交量',
            type: 'bar',
            xAxisIndex: 2,
            yAxisIndex: 2,
            data: volumeData,
            itemStyle: { color: '#1abc9c' },
            color: '#1abc9c'
          },
          {
            name: 'DIF',
            type: 'line',
            xAxisIndex: 3,
            yAxisIndex: 3,
            data: m.dif,
            smooth: false,
            showSymbol: false,
            lineStyle: { width: 1.2, color: '#e6a23c' },
            itemStyle: { color: '#e6a23c' }
          },
          {
            name: 'DEA',
            type: 'line',
            xAxisIndex: 3,
            yAxisIndex: 3,
            data: m.dea,
            smooth: false,
            showSymbol: false,
            lineStyle: { width: 1.2, color: '#409eff' },
            itemStyle: { color: '#409eff' }
          },
          {
            name: 'MACD',
            type: 'bar',
            xAxisIndex: 3,
            yAxisIndex: 3,
            data: macdData,
            // 图例方块用灰紫，柱子颜色仍由 macdData 里的红/绿 itemStyle 决定
            itemStyle: { color: '#8e44ad' },
            color: '#8e44ad'
          }
        ]
      };
    }

    /* 雪球圆形标：data/images/xueqiu_mark.png（构建时内联） */
    const XUEQIU_ICON_IMG = '__XUEQIU_DATA_URI__';
    const XUEQIU_ICON_SVG =
      '<img src="' + XUEQIU_ICON_IMG + '" width="14" height="14" alt="" draggable="false"/>';

    /* 当日热门板块：红色火炬（无「热门」文案） */
    const HOT_TORCH_SVG =
      '<svg class="hot-torch-icon" width="16" height="16" viewBox="0 0 24 24" aria-hidden="true" focusable="false">' +
      '<path fill="#c62828" d="M12 1.8c2.4 3.6.8 5.5-.6 7.2-1.4 1.7-2.4 2.9-2.4 4.8 0 2.1 1.5 3.6 3.5 4-.4-1.1-.2-2.1.5-3 .8-1.1 1.8-1.9 2.3-3.5.5-1.6.3-3.2-.6-4.8 2 1.7 3.1 4 3.1 6.4 0 4.6-3.3 7.8-7.2 7.8S3.8 17.5 3.8 12.9C3.8 8.4 7.5 4.6 12 1.8z"/>' +
      '<path fill="#ef5350" d="M12.1 6.2c.6 1.2.2 2.1-.5 2.9-.6.7-1.2 1.3-1.2 2.4 0 1 .7 1.7 1.6 2-.2-.6 0-1.1.4-1.6.5-.8 1.2-1.2 1.5-2.2.3-.9.2-1.8-.3-2.8.8.9 1.3 1.9 1.3 3.1 0 2.5-1.7 4.2-3.8 4.2S7.6 14.5 7.6 12c0-2.3 1.9-4.3 4.5-5.8z"/>' +
      '<path fill="#b71c1c" d="M10.4 18.2h3.2v2.2c0 .8-.6 1.4-1.4 1.4h-.4c-.8 0-1.4-.6-1.4-1.4v-2.2z"/>' +
      '</svg>';

    function hotTitle(s) {
      const tier = s.hotTier ? String(s.hotTier) : '热门';
      return '当日热门板块 · ' + tier + (s.industry ? ' · ' + s.industry : '');
    }

    function xueqiuUrl(code) {
      const prefix = code.charAt(0) === '6' ? 'SH' : 'SZ';
      return 'https://xueqiu.com/S/' + prefix + code;
    }

    function chgClass(v) {
      if (v == null || Number.isNaN(v)) return 'chg-flat';
      if (v > 0) return 'chg-up';
      if (v < 0) return 'chg-down';
      return 'chg-flat';
    }

    function fmtPct(v, digits) {
      if (v == null || Number.isNaN(v)) return '—';
      const sign = v > 0 ? '+' : '';
      return sign + v.toFixed(digits) + '%';
    }

    function fmtPrice(v) {
      if (v == null || Number.isNaN(v)) return '—';
      return String(v);
    }

    
    function adviceClass(advice) {
      if (!advice) return 'badge-advice-other';
      if (advice.indexOf('可买入') >= 0 || advice.indexOf('可短打') >= 0 || advice === '买入')
        return 'badge-advice-buy';
      if (advice.indexOf('观察') >= 0 || advice.indexOf('埋伏') >= 0 || advice === '观察')
        return 'badge-advice-watch';
      if (advice.indexOf('偏强') >= 0 || advice.indexOf('风险') >= 0) return 'badge-advice-hot';
      return 'badge-advice-other';
    }

    /** 页面展示用阶段文案（底层 stage 不变） */
    function displayAdvice(s) {
      const a = s.advice || '';
      if (a.indexOf('观察埋伏') >= 0) return '观察';
      if (a.indexOf('可短打') >= 0) return '买入';
      return a;
    }

    function escapeHtml(s) {
      return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    }

    function strategyClass(id) {
      const k = String(id || '').toLowerCase();
      if (k === 'strict') return 'badge-strategy-strict';
      if (k === 'r3') return 'badge-strategy-r3';
      if (k === 'scalp') return 'badge-strategy-scalp';
      if (k === 'dry_stable' || k === 'double_trough') return 'badge-strategy-' + k;
      if (k === 'expand_after_dry' || k === 'consol_vol_up') return 'badge-strategy-' + k;
      if (k === 'quiet_limit_up' || k === 'low_limit_up' || k === 'quiet_first') return 'badge-strategy-' + k;
      if (k === 'bottom_base') return 'badge-strategy-bottom_base';
      if (k === 'board_relaunch') return 'badge-strategy-board_relaunch';
      return 'badge-strategy-default';
    }
    function navStrategyClass(id) {
      const k = String(id || '').toLowerCase();
      if (k === 'strict') return 'nav-strategy nav-strategy-strict';
      if (k === 'r3') return 'nav-strategy nav-strategy-r3';
      if (k === 'scalp') return 'nav-strategy nav-strategy-scalp';
      if (k === 'dry_stable' || k === 'double_trough') return 'nav-strategy nav-strategy-' + k;
      if (k === 'expand_after_dry' || k === 'consol_vol_up') return 'nav-strategy nav-strategy-' + k;
      if (k === 'quiet_limit_up' || k === 'low_limit_up' || k === 'quiet_first') return 'nav-strategy nav-strategy-' + k;
      if (k === 'bottom_base') return 'nav-strategy nav-strategy-bottom_base';
      if (k === 'board_relaunch') return 'nav-strategy nav-strategy-board_relaunch';
      return 'nav-strategy nav-strategy-default';
    }
    /** 展示用策略标签：隐藏「默认」 */
    function visibleStrategies(s) {
      const tags = s.strategyTags || [];
      const ids = s.strategyIds || [];
      const out = [];
      for (let i = 0; i < tags.length; i++) {
        const id = String(ids[i] || tags[i] || '').toLowerCase();
        const title = String(tags[i] || '');
        if (id === 'default' || title === '默认') continue;
        out.push({ id: ids[i] || tags[i], title: title });
      }
      return out;
    }

    function farFromLowTitle(s) {
      const pct = s.dist60LowPct != null ? Number(s.dist60LowPct).toFixed(1) : '';
      const head = '距60日低点涨幅偏高（默认>20%）';
      if (pct) return head + '：当前 +' + pct + '%，底部发动段可能已过，慎追首打';
      return head + '，慎追首打';
    }

    function badgesHtml(s) {
      let html = '<span class="badges">';
      if (s.hotSector) {
        html += '<span class="badge badge-hot" title="' + escapeHtml(hotTitle(s)) + '">' + HOT_TORCH_SVG + '</span>';
      }
      if (s.warnFarFromLow) {
        html += '<span class="badge badge-warn-far-low" title="' + escapeHtml(farFromLowTitle(s)) + '">离底远</span>';
      }
      if (s.industry)
        html += '<span class="badge badge-industry' + (s.hotSector ? ' badge-industry-hot' : '') + '">' +
          escapeHtml(s.industry) + '</span>';
      const adviceShow = displayAdvice(s);
      if (adviceShow)
        html += '<span class="badge badge-advice ' + adviceClass(s.advice || adviceShow) + '">' + escapeHtml(adviceShow) + '</span>';
      html += tipHtml(s);
      html += '</span>';
      const strats = visibleStrategies(s);
      if (strats.length) {
        html += '<span class="badges badges-strategy">';
        for (let i = 0; i < strats.length; i++) {
          html += '<span class="badge badge-strategy ' + strategyClass(strats[i].id) + '">' + escapeHtml(strats[i].title) + '</span>';
        }
        html += '</span>';
      }
      return html;
    }

    function tipHtml(s) {
      const a = s.advice || '';
      let label = '';
      if (a.indexOf('观察') >= 0 || a.indexOf('埋伏') >= 0) label = '进场条件';
      else if (a.indexOf('可买入') >= 0 || a.indexOf('可短打') >= 0) label = '离场条件';
      else return '';
      const text = (s.when || '').trim();
      if (!text) return '';
      return '<span class="cond-tip"><span class="cond-label">' + label + '</span> ' + escapeHtml(text) + '</span>';
    }

    function metaHtml(s) {
      const parts = [];
      function chip(key, val, extraClass) {
        const cls = extraClass ? ('metric ' + extraClass) : 'metric';
        return (
          '<span class="' + cls + '">' +
          '<span class="metric-k">' + key + '</span>' +
          '<span class="metric-v">' + val + '</span>' +
          '</span>'
        );
      }
      if (s.scoreEntry != null)
        parts.push(chip('Entry', Number(s.scoreEntry).toFixed(3), 'metric-score'));
      if (s.scoreSetup != null)
        parts.push(chip('Setup', Number(s.scoreSetup).toFixed(3), 'metric-score'));
      if (s.score != null)
        parts.push(chip('画像', Number(s.score).toFixed(3), 'metric-score'));
      if (s.hardScore != null)
        parts.push(chip('硬条件', Number(s.hardScore).toFixed(3), 'metric-score'));
      if (s.dist60LowPct != null)
        parts.push(chip('距60低', '+' + Number(s.dist60LowPct).toFixed(1) + '%', s.warnFarFromLow ? 'metric-warn-far-low' : 'metric-dist60'));
      if (s.boxBottom != null)
        parts.push(chip('箱体底', fmtPrice(s.boxBottom), 'metric-box'));
      if (s.takeProfit != null)
        parts.push(chip('止盈', fmtPrice(s.takeProfit), 'metric-tp'));
      parts.push(chip('收盘', fmtPrice(s.close), chgClass(s.ret1d)));
      parts.push(chip('今日', fmtPct(s.ret1d, 2), chgClass(s.ret1d)));
      if (s.ret60 != null)
        parts.push(chip('60日', fmtPct(s.ret60, 1), chgClass(s.ret60)));
      if (s.turnover != null)
        parts.push(chip('换手', s.turnover.toFixed(2) + '%', 'metric-turn'));
      if (s.amountYi != null)
        parts.push(chip('成交额', s.amountYi.toFixed(4) + '亿', 'metric-amt'));
      if (s.volumeYi != null)
        parts.push(chip('成交量', s.volumeYi.toFixed(4) + '亿股', 'metric-vol'));
      if (s.floatYi != null)
        parts.push(chip('流通', s.floatYi.toFixed(4) + '亿股', 'metric-float'));
      if (s.bars && s.bars.length)
        parts.push(chip('K线', String(s.bars.length) + ' 根', 'metric-bars'));
      return '<span class="meta">' + parts.join('') + '</span>';
    }


    function isBuy(s) {
      if (s.canBuy === true) return true;
      const a = s.advice || '';
      return a.indexOf('可买入') >= 0 || a.indexOf('可短打') >= 0;
    }
    function isWatch(s) {
      const a = s.advice || '';
      return a.indexOf('观察埋伏') >= 0 || a.indexOf('宝藏观察') >= 0 || a === '观察';
    }
    function buyLabel(s) {
      const a = s.advice || '';
      if (typeof currentTab !== 'undefined' && currentTab === 'board') return '续板';
      if (a.indexOf('可短打') >= 0) return '买入';
      if (a.indexOf('可买入') >= 0) return '可买';
      return '买入';
    }
    function watchLabel(s) {
      const a = s.advice || '';
      if (a.indexOf('宝藏') >= 0) return '宝藏';
      return '观察';
    }
    const chartInstances = {};
    const panelBuilt = { long: false, short: false, treasure: false, board: false, base: false, relaunch: false };
    let panesReady = false;

    function disposeCharts() {
      Object.keys(chartInstances).forEach(sid => disposeChart(sid));
      charts = [];
    }
    function disposeChart(sid) {
      const c = chartInstances[sid];
      if (!c) return;
      try { c.dispose(); } catch (e) {}
      delete chartInstances[sid];
      charts = Object.values(chartInstances);
    }
    function activeStocks() {
      if (typeof TABBED !== 'undefined' && TABBED) return PANELS[currentTab] || [];
      return STOCKS || [];
    }
    function updateHeaderCounts(tab) {
      const list = (typeof TABBED !== 'undefined' && TABBED)
        ? (PANELS[tab] || [])
        : (STOCKS || []);
      const buyN = list.filter(isBuy).length;
      const countEl = document.getElementById('count');
      if (countEl) countEl.textContent = String(list.length);
      const buyEl = document.getElementById('buy-count');
      if (buyEl) buyEl.textContent = String(buyN);
      const labelEl = document.getElementById('tab-label');
      if (labelEl) labelEl.textContent = (TAB_LABEL && TAB_LABEL[tab]) || '';
      if (typeof TABBED !== 'undefined' && TABBED) {
        const nl = document.getElementById('n-long');
        const nt = document.getElementById('n-treasure');
        const ns = document.getElementById('n-short');
        const np = document.getElementById('n-scalp');
        const nb = document.getElementById('n-board');
        const nbase = document.getElementById('n-base');
        const nr = document.getElementById('n-relaunch');
        if (nl) nl.textContent = '(' + String((PANELS.long || []).length) + ')';
        if (nt) nt.textContent = '(' + String((PANELS.treasure || []).length) + ')';
        if (ns) ns.textContent = '(' + String((PANELS.short || []).length) + ')';
        if (np) np.textContent = '(' + String((PANELS.scalp || []).length) + ')';
        if (nb) nb.textContent = '(' + String((PANELS.board || []).length) + ')';
        if (nbase) nbase.textContent = '(' + String((PANELS.base || []).length) + ')';
        if (nr) nr.textContent = '(' + String((PANELS.relaunch || []).length) + ')';
      }
    }
    function ensureTabPanes() {
      if (panesReady || typeof TABBED === 'undefined' || !TABBED) return;
      const nav = document.getElementById('nav');
      const main = document.getElementById('main');
      nav.innerHTML =
        '<div id="nav-long" class="tab-pane"></div>' +
        '<div id="nav-short" class="tab-pane" hidden></div>' +
        '<div id="nav-scalp" class="tab-pane" hidden></div>' +
        '<div id="nav-treasure" class="tab-pane" hidden></div>' +
        '<div id="nav-board" class="tab-pane" hidden></div>' +
        '<div id="nav-base" class="tab-pane" hidden></div>' +
        '<div id="nav-relaunch" class="tab-pane" hidden></div>';
      main.innerHTML =
        '<div id="main-long" class="tab-pane"></div>' +
        '<div id="main-short" class="tab-pane" hidden></div>' +
        '<div id="main-scalp" class="tab-pane" hidden></div>' +
        '<div id="main-treasure" class="tab-pane" hidden></div>' +
        '<div id="main-board" class="tab-pane" hidden></div>' +
        '<div id="main-base" class="tab-pane" hidden></div>' +
        '<div id="main-relaunch" class="tab-pane" hidden></div>';
      panesReady = true;
    }
    function resizeTabCharts(tab) {
      const prefix = tab + '-';
      Object.keys(chartInstances).forEach(sid => {
        if (!sid.startsWith(prefix)) return;
        try { chartInstances[sid].resize(); } catch (e) {}
      });
    }
    function showTab(tab) {
      const prev = currentTab;
      currentTab = tab;
      document.querySelectorAll('.tab-btn').forEach(b => {
        b.classList.toggle('active', b.getAttribute('data-tab') === tab);
      });
      updateHeaderCounts(tab);
      ['long', 'short', 'scalp', 'treasure', 'board', 'base', 'relaunch'].forEach(t => {
        const hide = t !== tab;
        const n = document.getElementById('nav-' + t);
        const m = document.getElementById('main-' + t);
        if (n) n.hidden = hide;
        if (m) {
          m.hidden = hide;
          m.classList.remove('tab-enter');
          if (!hide) {
            // 强制重启动效
            void m.offsetWidth;
            m.classList.add('tab-enter');
          }
        }
      });
      buildPanel(tab);
      requestAnimationFrame(() => resizeTabCharts(tab));
      void prev;
    }
    function buildPanel(tab) {
      if (typeof TABBED !== 'undefined' && TABBED && panelBuilt[tab]) return;
      const list = (typeof TABBED !== 'undefined' && TABBED)
        ? (PANELS[tab] || [])
        : (STOCKS || []);
      const navRoot = (typeof TABBED !== 'undefined' && TABBED)
        ? document.getElementById('nav-' + tab)
        : document.getElementById('nav');
      const mainRoot = (typeof TABBED !== 'undefined' && TABBED)
        ? document.getElementById('main-' + tab)
        : document.getElementById('main');
      if (!navRoot || !mainRoot) return;
      navRoot.innerHTML = '';
      mainRoot.innerHTML = '';
      const prefix = (typeof TABBED !== 'undefined' && TABBED) ? (tab + '-') : '';
      list.forEach((s) => {
        const sid = prefix + s.code;
        const item = document.createElement('div');
        item.className = 'nav-item';

        const rowMain = document.createElement('div');
        rowMain.className = 'nav-row nav-row-main';
        const a = document.createElement('a');
        a.className = isBuy(s) ? 'anchor buy' : 'anchor';
        a.href = '#' + sid;
        a.textContent = s.code + ' ' + s.name;
        rowMain.appendChild(a);
        if (s.industry) {
          const ind = document.createElement('span');
          ind.className = s.hotSector ? 'nav-industry nav-industry-hot' : 'nav-industry';
          ind.textContent = s.industry;
          ind.title = s.hotSector
            ? ('当日热门 · ' + (s.hotTier || '') + ' · ' + s.industry)
            : s.industry;
          rowMain.appendChild(ind);
        }
        const xq = document.createElement('a');
        xq.className = 'xq-link';
        xq.href = xueqiuUrl(s.code);
        xq.target = '_blank';
        xq.rel = 'noopener noreferrer';
        xq.title = '在雪球打开 ' + s.code + ' ' + s.name;
        xq.setAttribute('aria-label', '雪球 ' + s.code);
        xq.innerHTML = XUEQIU_ICON_SVG;
        rowMain.appendChild(xq);
        item.appendChild(rowMain);

        const rowTags = document.createElement('div');
        rowTags.className = 'nav-row nav-row-tags';
        let hasTag = false;
        if (isBuy(s)) {
          const tag = document.createElement('span');
          tag.className = 'nav-buy';
          tag.textContent = buyLabel(s);
          rowTags.appendChild(tag);
          hasTag = true;
        } else if (isWatch(s)) {
          const tag = document.createElement('span');
          tag.className = 'nav-watch';
          tag.textContent = watchLabel(s);
          rowTags.appendChild(tag);
          hasTag = true;
        }
        if (s.hotSector) {
          const hot = document.createElement('span');
          hot.className = 'nav-hot';
          hot.title = hotTitle(s);
          hot.setAttribute('aria-label', hotTitle(s));
          hot.innerHTML = HOT_TORCH_SVG;
          rowTags.appendChild(hot);
          hasTag = true;
        }
        if (s.warnFarFromLow) {
          const warn = document.createElement('span');
          warn.className = 'nav-warn-far-low';
          warn.textContent = '离底远';
          warn.title = farFromLowTitle(s);
          rowTags.appendChild(warn);
          hasTag = true;
        }
        const strats = visibleStrategies(s);
        for (let i = 0; i < strats.length; i++) {
          const st = document.createElement('span');
          st.className = navStrategyClass(strats[i].id);
          st.textContent = strats[i].title;
          st.title = strats[i].title;
          rowTags.appendChild(st);
          hasTag = true;
        }
        if (hasTag) item.appendChild(rowTags);

        navRoot.appendChild(item);
        const section = document.createElement('section');
        section.className = 'card';
        section.id = sid;
        section.innerHTML =
          '<div class="card-head"><h2>' + escapeHtml(s.code + ' ' + s.name) + '</h2>' +
          badgesHtml(s) +
          metaHtml(s) + '</div>' +
          '<div class="chart-wrap open" id="chart-wrap-' + sid + '">' +
          (s.bars && s.bars.length
            ? '<div class="chart" id="chart-' + sid + '"></div>'
            : '<p class="chart-error">无K线数据</p>') +
          '</div>';
        mainRoot.appendChild(section);
        if (s.bars && s.bars.length) {
          const el = document.getElementById('chart-' + sid);
          const chart = echarts.init(el, null, { renderer: 'canvas' });
          chart.setOption(buildOption(s, tab));
          chartInstances[sid] = chart;
        }
      });
      charts = Object.values(chartInstances);
      if (typeof TABBED !== 'undefined' && TABBED) panelBuilt[tab] = true;
    }
    function render() {
      if (typeof TABBED !== 'undefined' && TABBED) {
        ensureTabPanes();
        updateHeaderCounts(currentTab);
        const nl = document.getElementById('n-long');
        const nt = document.getElementById('n-treasure');
        const ns = document.getElementById('n-short');
        const np = document.getElementById('n-scalp');
        const nb = document.getElementById('n-board');
        const nbase = document.getElementById('n-base');
        const nr = document.getElementById('n-relaunch');
        if (nl) nl.textContent = '(' + String((PANELS.long || []).length) + ')';
        if (nt) nt.textContent = '(' + String((PANELS.treasure || []).length) + ')';
        if (ns) ns.textContent = '(' + String((PANELS.short || []).length) + ')';
        if (np) np.textContent = '(' + String((PANELS.scalp || []).length) + ')';
        if (nb) nb.textContent = '(' + String((PANELS.board || []).length) + ')';
        if (nbase) nbase.textContent = '(' + String((PANELS.base || []).length) + ')';
        if (nr) nr.textContent = '(' + String((PANELS.relaunch || []).length) + ')';
        showTab(currentTab);
        return;
      }
      disposeCharts();
      updateHeaderCounts('long');
      buildPanel('long');
    }
    function syncHeaderOffset() {
      const header = document.querySelector('.content > header');
      if (!header) return;
      const h = Math.ceil(header.getBoundingClientRect().height);
      if (h > 0) {
        document.documentElement.style.setProperty('--sticky-header-h', h + 'px');
      }
    }
    function scrollToCard(sid) {
      const el = document.getElementById(sid);
      if (!el) return;
      syncHeaderOffset();
      const header = document.querySelector('.content > header');
      const offset = (header ? header.getBoundingClientRect().height : 120) + 10;
      const top = el.getBoundingClientRect().top + window.scrollY - offset;
      window.scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
    }
    function bindNavAnchors() {
      const nav = document.getElementById('nav');
      if (!nav || nav.dataset.anchorBound === '1') return;
      nav.dataset.anchorBound = '1';
      nav.addEventListener('click', (ev) => {
        const a = ev.target.closest('a.anchor');
        if (!a) return;
        const href = a.getAttribute('href') || '';
        if (!href.startsWith('#')) return;
        const sid = href.slice(1);
        if (!sid || !document.getElementById(sid)) return;
        ev.preventDefault();
        if (history.replaceState) {
          history.replaceState(null, '', href);
        } else {
          location.hash = sid;
        }
        scrollToCard(sid);
      });
    }
    function bindTabs() {
      if (typeof TABBED === 'undefined' || !TABBED) return;
      document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          const tab = btn.getAttribute('data-tab');
          if (!tab || tab === currentTab) return;
          showTab(tab);
          syncHeaderOffset();
          window.scrollTo({ top: 0, behavior: 'smooth' });
        });
      });
    }
