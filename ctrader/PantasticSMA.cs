using cAlgo.API;
using cAlgo.API.Indicators;

namespace cAlgo.Indicators
{
    [Indicator(IsOverlay = true, AutoRescale = false, AccessRights = AccessRights.None)]
    public class PantasticSMA : Indicator
    {
        [Parameter("Source")]
        public DataSeries Source { get; set; }

        [Parameter("SMA Periods", DefaultValue = 7)]
        public int SmaPeriods { get; set; }

        [Parameter("Momentum Lookback", DefaultValue = 5)]
        public int MomentumLookback { get; set; }

        [Parameter("Momentum Threshold Pips/Min", DefaultValue = 0.003)]
        public double MomentumThresholdPipsPerMin { get; set; }

        [Parameter("Momentum Threshold Pips/Bar", DefaultValue = 0.003)]
        public double MomentumThresholdPipsPerBar { get; set; }

        // false → pips/bar threshold | true → pips/min (timeframe-neutral)
        [Parameter("White Method", DefaultValue = false)]
        public bool WhiteMethod { get; set; }

        // Declared so the .indiset "SmaResult" line name resolves; never populated
        // so it draws nothing — all rendering is via Chart.DrawTrendLine segments
        [Output("SmaResult", LineColor = "Transparent", PlotType = PlotType.Line, Thickness = 1)]
        public IndicatorDataSeries SmaResult { get; set; }

        private MovingAverage _sma;
        private double _barMinutes;

        protected override void Initialize()
        {
            _sma = Indicators.MovingAverage(Source, SmaPeriods, MovingAverageType.Simple);
            _barMinutes = Bars.Count >= 2
                ? (Bars.Last(0).OpenTime - Bars.Last(1).OpenTime).TotalMinutes
                : 1.0;
        }

        public override void Calculate(int index)
        {
            double sma = _sma.Result[index];
            // SmaResult intentionally left as NaN — DrawTrendLine handles all rendering

            if (index < MomentumLookback + 1)
                return;

            // Raw price change over the lookback window (not converted to pips)
            // Thresholds are in price units/bar or price units/min — 0.003 on
            // GBP/JPY ≈ 0.3 pips/bar, high enough that only strong moves color
            double priceChange = sma - _sma.Result[index - MomentumLookback];

            double rate = WhiteMethod
                ? priceChange / (MomentumLookback * _barMinutes)  // price per minute
                : priceChange /  MomentumLookback;                 // price per bar

            double threshold = WhiteMethod
                ? MomentumThresholdPipsPerMin
                : MomentumThresholdPipsPerBar;

            Color segColor;
            if      (rate >  threshold) segColor = Color.Red;
            else if (rate < -threshold) segColor = Color.DodgerBlue;
            else                        segColor = Color.White;

            // Draw one segment from previous bar to current bar.
            // Unique name per bar index → recalculation overwrites cleanly.
            Chart.DrawTrendLine(
                "psma_" + index,
                Bars[index - 1].OpenTime, _sma.Result[index - 1],
                Bars[index].OpenTime,     sma,
                segColor, 1, LineStyle.Solid);
        }
    }
}
