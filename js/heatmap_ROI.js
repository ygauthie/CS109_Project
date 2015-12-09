
/**
 * Grid-light theme for Highcharts JS
 * @author Torstein Honsi
 */

// Load the fonts
Highcharts.createElement('link', {
   href: '//fonts.googleapis.com/css?family=Roboto+Slab:400,600',
   rel: 'stylesheet',
   type: 'text/css'
}, null, document.getElementsByTagName('head')[0]);

Highcharts.theme = {
   colors: ["#7cb5ec", "#f7a35c", "#90ee7e", "#7798BF", "#aaeeee", "#ff0066", "#eeaaee",
      "#55BF3B", "#DF5353", "#7798BF", "#aaeeee"],
   chart: {
      backgroundColor: null,
      style: {
         fontFamily: "Roboto Slab, sans-serif"
      }
   },
   title: {
      style: {
         fontSize: '16px',
         fontWeight: 'bold',
         textTransform: 'uppercase'
      }
   },
   tooltip: {
      borderWidth: 0,
      backgroundColor: 'rgba(219,219,216,0.8)',
      shadow: false
   },
   legend: {
      itemStyle: {
         fontWeight: 'bold',
         fontSize: '13px'
      }
   },
   xAxis: {
      gridLineWidth: 1,
      labels: {
         style: {
            fontSize: '12px'
         }
      }
   },
   yAxis: {
      minorTickInterval: 'auto',
      title: {
         style: {
            textTransform: 'uppercase'
         }
      },
      labels: {
         style: {
            fontSize: '12px'
         }
      }
   },
   plotOptions: {
      candlestick: {
         lineColor: '#404048'
      }
   },


   // General
   background2: '#F0F0EA'
   
};

// Apply the theme
Highcharts.setOptions(Highcharts.theme);

$(function () {

    $('#ROI_container').highcharts({

        chart: {
            type: 'heatmap',
            marginTop: 60,
            marginBottom: 80,
            plotBorderWidth: 1
        },


        title: {
            text: 'Return on Investment (ROI)'
        },

        subtitle: {
            text: 'Simulated trading from 01 Jan to 01 Dec 2015'
        },

        xAxis: {
            categories: ['SVM (RBF)', 'Extra-trees', 'Gaussian Naive Bayes', 'Logistic regression', 'Random forest', 'Ensemble']
        },

        yAxis: {
            categories: ['Energy (IYE)', 'Financials (IYF)', 'Materials (IYM)', 'Home building (ITB)', 'Healthcare (IYH)', 'Industrials (IYJ)', 'Technology (IYW)', 'Real estate (IYR)', 'Telecoms (IYZ)'],
            title: null
        },

        colorAxis: {
            reversed: false,
            min: -0.15,
            max:0.15,
            stops: [
                [0, '#d7191c'],
                [0.25, '#fdae61'],
                [0.5, '#ffffff'],
                [0.75, '#abd9e9'],
                [1, '#2c7bb6']
            ],
            labels: {
                    formatter: function() {
                        return this.value * 100 + '%'; }
            }
        },

        legend: {
            align: 'right',
            layout: 'vertical',
            margin: 0,
            verticalAlign: 'top',
            y: 35,
            symbolHeight: 280
        },

        tooltip: {
            formatter: function () {
                return 'Trading <b>' + this.series.yAxis.categories[this.point.y] + '</b> based on predictions from <br> the <b>' + this.series.xAxis.categories[this.point.x] + '</b> classifier would have returned <b>' +
                    Math.round(this.point.value*100) + '</b>% <br>on investment.';
            }
        },

        series: [{
            name: 'Accuracy',
            borderWidth: 1,
            data: [[0, 0, -0.06], [1, 0, -0.08], [2, 0, -0.07], [3, 0, 0.02], [4, 0, -0.03], [5, 0, -0.06], [0, 1, 0.07], [1, 1, 0.1], [2, 1, 0.08], [3, 1, 0.08], [4, 1, 0.06], [5, 1, 0.08], [0, 2, -0.06], [1, 2, -0.08], [2, 2, -0.17], [3, 2, -0.06], [4, 2, -0.16], [5, 2, -0.09], [0, 3, 0.19], [1, 3, 0.24], [2, 3, 0.28], [3, 3, 0.31], [4, 3, 0.2], [5, 3, 0.28], [0, 4, 0.13], [1, 4, 0.14], [2, 4, 0.21], [3, 4, 0.07], [4, 4, 0.17], [5, 4, 0.13], [0, 5, 0.09], [1, 5, 0.13], [2, 5, 0.06], [3, 5, 0.02], [4, 5, 0.05], [5, 5, 0.12], [0, 6, 0.1], [1, 6, 0.08], [2, 6, 0.1], [3, 6, 0.07], [4, 6, 0.2], [5, 6, 0.15], [0, 7, -0.05], [1, 7, -0.07], [2, 7, -0.01], [3, 7, -0.06], [4, 7, -0.05], [5, 7, -0.03], [0, 8, 0.15], [1, 8, 0.18], [2, 8, 0.09], [3, 8, 0.07], [4, 8, 0.1], [5, 8, 0.15]],
                dataLabels: {
                enabled: true,
                color: '#000000',
                formatter: function () { return Math.round(this.point.value*100) + '%'; }
                //format: '{point.value}'
            }
        }]

    });
});
