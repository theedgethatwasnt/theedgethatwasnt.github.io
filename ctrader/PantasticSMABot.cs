using cAlgo.API;
using cAlgo.API.Indicators;

namespace cAlgo.Robots
{
    [Robot(AccessRights = AccessRights.None)]
    public class PantasticSMABot : Robot
    {
        [Parameter(DefaultValue = 100000)]
        public double Volume { get; set; }

        [Parameter("SMA Periods", DefaultValue = 7)]
        public int SmaPeriods { get; set; }

        [Parameter("Momentum Lookback", DefaultValue = 5)]
        public int MomentumLookback { get; set; }

        [Parameter("Momentum Threshold (price/min)", DefaultValue = 0.003)]
        public double MomentumThresholdPerMin { get; set; }

        [Parameter("Momentum Threshold (price/bar)", DefaultValue = 0.003)]
        public double MomentumThresholdPerBar { get; set; }

        // false → price/bar threshold | true → price/min threshold
        [Parameter("White Method (use price/min)", DefaultValue = false)]
        public bool WhiteMethod { get; set; }

        private MovingAverage _sma;
        private double _barMinutes;

        protected override void OnStart()
        {
            _sma = Indicators.MovingAverage(Bars.ClosePrices, SmaPeriods, MovingAverageType.Simple);
            _barMinutes = Bars.Count >= 2
                ? (Bars.Last(0).OpenTime - Bars.Last(1).OpenTime).TotalMinutes
                : 1.0;
        }

        protected override void OnBarClosed()
        {
            int index = Bars.Count - 1;

            if (index < MomentumLookback + 1)
                return;

            double sma      = _sma.Result.Last(0);
            double smaPrev  = _sma.Result[index - MomentumLookback];
            double priceChange = sma - smaPrev;

            double rate = WhiteMethod
                ? priceChange / (MomentumLookback * _barMinutes)  // price per minute
                : priceChange /  MomentumLookback;                 // price per bar

            double threshold = WhiteMethod
                ? MomentumThresholdPerMin
                : MomentumThresholdPerBar;

            if      (rate >  threshold) GoLong();
            else if (rate < -threshold) GoShort();
            else                        Flatten();
        }

        private void GoLong()
        {
            foreach (var pos in Positions.FindAll(InstanceId, SymbolName, TradeType.Sell))
                ClosePosition(pos);

            if (Positions.FindAll(InstanceId, SymbolName, TradeType.Buy).Length == 0)
                ExecuteMarketOrder(TradeType.Buy, SymbolName, Volume, InstanceId);
        }

        private void GoShort()
        {
            foreach (var pos in Positions.FindAll(InstanceId, SymbolName, TradeType.Buy))
                ClosePosition(pos);

            if (Positions.FindAll(InstanceId, SymbolName, TradeType.Sell).Length == 0)
                ExecuteMarketOrder(TradeType.Sell, SymbolName, Volume, InstanceId);
        }

        private void Flatten()
        {
            foreach (var pos in Positions.FindAll(InstanceId, SymbolName))
                ClosePosition(pos);
        }

        protected override void OnStop()
        {
            Flatten();
        }
    }
}
