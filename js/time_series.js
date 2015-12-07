
$(function () {
    var seriesOptions = [],
        seriesCounter = 0,
        names = ['Index','Ensemble'],
        //names = ['Index','Ensemble','Extra_Trees','Gaussian_NB','log_regression','Random_forest','RBF_SVM'],
        tickers = ['IYJ','IYF','IYW','IYZ','ITB','IYE','IYH','IYM','IYR'];
    /**
     * Create the chart when all data is loaded
     * @returns {undefined}
     */
    function createChart(ticker) {

        $('#'+ticker+'_container').highcharts('StockChart', {
  
            chart: {
                marginTop: 70
            },

            rangeSelector: {
                selected: 4
            },

            title: {
                text: 'Simulated growth of a $10,000 investment'
            },

            subtitle: {
                text: 'when trading based on ensemble classifier predictions'
            },
 
            yAxis: {
                labels: {
                    formatter: function () {
                        return (this.value > 0 ? ' + ' : '') + this.value + '%';
                    }
                },
                plotLines: [{
                    value: 0,
                    width: 1,
                    color: 'silver'
                }]
            },
             
            legend: {
                align: 'left',
                verticalAlign: 'center',
                layout: 'vertical',
                enabled: true
            },

            plotOptions: {
                series: {
                    compare: 'percent'
                }
            },

            tooltip: {
                pointFormat: '<span style="color:{series.color}">{series.name}</span>: <b>{point.y}</b> ({point.change}%)<br/>',
                valueDecimals: 2
            },

            series: seriesOptions
        });
    }
    // Set the global configs to synchronous 
    $.ajaxSetup({
        async: false
    });

    $.each(tickers, function (j, ticker) {
      $.each(names, function (i, name) {
        $.getJSON('./data/time_series/'+ticker + '_' + name + '.json',    function (data) {
            seriesOptions[i] = {
                name: name,
                data: data
            };
            
            // As we're loading the data asynchronously, we don't know what order it will arrive. So
            // we keep a counter and create the chart when all the data is loaded.
            seriesCounter += 1;
            if (i === names.length - 1) {
                createChart(ticker);
                seriesCounter = 0;
            }
        });
      });
    });
});