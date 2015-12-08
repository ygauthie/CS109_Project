
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

    $('#accuracy_container').highcharts({

        chart: {
            type: 'heatmap',
            marginTop: 60,
            marginBottom: 80,
            plotBorderWidth: 1
        },


        title: {
            text: 'Model accuracies on test set'
        },

        subtitle: {
            text: '(trading days from 01 Jan to 01 Dec 2015)'
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
            min: 0.61,
            max:0.75,
            stops: [
                [0, '#ffffff'],
                [0.5, '#abd9e9'],
                [1, '#2c7bb6']
            ],
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
                return 'The <b>' + this.series.xAxis.categories[this.point.x] + '</b> classifier was accurate <br><b>' +
                    Math.round(this.point.value*100)+ '</b>% of trading days for <b>' + this.series.yAxis.categories[this.point.y] + '</b>';
            }
        },

        series: [{
            name: 'Accuracy',
            borderWidth: 1,
            data: [[0, 0, 0.66], [1, 0, 0.64], [2, 0, 0.63], [3, 0, 0.67], [4, 0, 0.62], [5, 0, 0.65], [0, 1, 0.72], [1, 1, 0.71], [2, 1, 0.72], [3, 1, 0.71], [4, 1, 0.72], [5, 1, 0.73], [0, 2, 0.69], [1, 2, 0.67], [2, 2, 0.64], [3, 2, 0.68], [4, 2, 0.62], [5, 2, 0.68], [0, 3, 0.62], [1, 3, 0.68], [2, 3, 0.67], [3, 3, 0.67], [4, 3, 0.66], [5, 3, 0.68], [0, 4, 0.7], [1, 4, 0.7], [2, 4, 0.75], [3, 4, 0.67], [4, 4, 0.73], [5, 4, 0.71], [0, 5, 0.73], [1, 5, 0.74], [2, 5, 0.74], [3, 5, 0.7], [4, 5, 0.74], [5, 5, 0.74], [0, 6, 0.73], [1, 6, 0.69], [2, 6, 0.72], [3, 6, 0.68], [4, 6, 0.72], [5, 6, 0.73], [0, 7, 0.71], [1, 7, 0.68], [2, 7, 0.69], [3, 7, 0.67], [4, 7, 0.71], [5, 7, 0.7], [0, 8, 0.74], [1, 8, 0.71], [2, 8, 0.68], [3, 8, 0.68], [4, 8, 0.68], [5, 8, 0.72]],
                dataLabels: {
                enabled: true,
                color: '#000000'
            }
        }]

    });
});
