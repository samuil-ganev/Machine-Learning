import java.util.ArrayList;

class Functions {
  
  public static double yMax = 1;

  /**

    Scaler

  */

  static double[] scaler(double[] data) {

    double min = 10_000, max = -10_000;

    for (int i=0;i<data.length;++i) {

      if (data[i] < min) {

        min = data[i];

      } else if (data[i] > max) {

        max = data[i];

      }
      
    }
    
    yMax = max;

    for (int i=0;i<data.length;++i) {

      data[i] = (data[i] - min) / (max - min);

    }

    return data;

  }

  static double[][] scaler(double[][] data) {

    int n = data.length, m = data[0].length;

    for (int i=0;i<m;++i) {

      double min = 10_000, max = -10_000;

      for (int j=0;j<n;++j) {

        if (data[j][i] < min) {

          min = data[j][i];

        } else if (data[j][i] > max) {

          max = data[j][i];

        }

      }

      for (int j=0;j<n;++j) {

        data[j][i] = (data[j][i] - min) / (max - min);

      }

    }

    return data;

  }

  static double normalDistribution(int x) {

    return Math.pow(Math.E, -Math.pow(x, 2) / 2) / Math.sqrt(2 * Math.PI);

  }

  /**

    Activation Functions

  */

  static double softplus(double x) {

    return Math.log(1 + Math.pow(Math.E, x));

  }

  static double sigmoid(double x) {

    return 1 / (1 + Math.pow(Math.E, -x));

  }

  static double relu(double x) {

    return Math.max(0, x);

  }

  /**

    Loss functions

  */

  static double crossEntropy(double yPredicted, double y) {

    return y * Math.log(yPredicted);

  }

  static double MSE(double[] yPredicted, double[] y) {

    double sum = 0;
    int n = y.length;

    for (int i=0;i<n;++i)
      sum += Math.pow((yPredicted[i] - y[i]), 2);

    return sum / n;

  }

  /**

    Manipulate data

  */

  static Pair getLayerOfNode(int number, int[][] connections) {

    for (int i=0;i<connections.length;++i) {

      for (int j=0;j<connections[0].length;++j) {

        if (connections[i][j] == -1)
          break;

        if (connections[i][j] == number)
          return new Pair(i, j);

      }

    }

    return new Pair(-1, -1);

  }

  static boolean isPartOf(ArrayList<Double> array, double element) {

    for (int i=0;true;++i) {

      try {

        if (array.get(i) == element)
          return true;

      } catch (Exception e) { return false; }

    }

  }

  static int unique(double[] labels) {

    ArrayList<Double> set = new ArrayList<Double>();

    for (double label : labels) {

      if (!isPartOf(set, label))
        set.add(label);

    }

    return set.size();

  }

  static boolean alreadyCovered(ArrayList<Pair> covered, Pair cover) {

    for (int i=0;i<covered.size();++i) {

      if (covered.get(i).getKey() == cover.key && covered.get(i).getValue() == cover.value || covered.get(i).getKey() == cover.value && covered.get(i).getValue() == cover.key)
        return true;

    }

    return false;

  }

  /**

    Derivative Calculator


  */

  static double dMSE(int n, double yPredicted, double y) {

    return 2.0 / n * (yPredicted - y);

  }

  static double dsigmoid(double x) {

    return sigmoid(x) * (1 - sigmoid(x));

  }

  static double drelu(double x) {

    if (x < 0)
      return 0;

    return 1;

  }

  static double dsoftplus(double x) {

    return sigmoid(x);

  }

  static double calculateDerivativeW(Pair coordinates, double[][] weights, double[][][] values, double[][][] valuesBeforeActivations, int[][] connections, double[] yPredicted, double[] y, Dense[] denses) {

      double derivative = 0;

      for (int i=0;i<values.length;++i) {

        double sampleDerivative = dMSE(y.length, yPredicted[i], y[i]);

        Pair nodeCoordinates = getLayerOfNode(coordinates.key, connections);
        Pair secondNodeCoordinates = getLayerOfNode(coordinates.value, connections);

        for (int layer=values[i].length-1;layer>nodeCoordinates.key;layer--) {

          String activation = denses[layer].getActivation();

          switch (activation) {

            case "sigmoid":
              sampleDerivative *= dsigmoid(valuesBeforeActivations[i][layer][0]);
              break;
            case "softplus":
              sampleDerivative *= dsoftplus(valuesBeforeActivations[i][layer][0]);
              break;
            case "relu":
              sampleDerivative *= drelu(valuesBeforeActivations[i][layer][0]);
              break;
            default:
              break;

          }

          if (layer != nodeCoordinates.key + 1) {
            
            sampleDerivative *= weights[connections[layer][0]][connections[layer-1][0]];

          } else {

            sampleDerivative *= weights[connections[layer][0]][connections[layer-1][nodeCoordinates.value]];

          }

        }

       // sampleDerivative *= weights[coordinates.key][coordinates.value];
        derivative += sampleDerivative;

      }

    return derivative;

  }
  
static double calculateDerivativeB(Pair coordinates, double[][] weights, double[][][] values, double[][][] valuesBeforeActivations, int[][] connections, double[] yPredicted, double[] y, Dense[] denses) {

      double derivative = 0;

      for (int i=0;i<values.length;++i) {

        double sampleDerivative = dMSE(y.length, yPredicted[i], y[i]);

        for (int layer=values[i].length-1;layer>coordinates.key;layer--) {

          String activation = denses[layer].getActivation();

          switch (activation) {

            case "sigmoid":
              sampleDerivative *= dsigmoid(valuesBeforeActivations[i][layer][0]);
              break;
            case "softplus":
              sampleDerivative *= dsoftplus(valuesBeforeActivations[i][layer][0]);
              break;
            case "relu":
              sampleDerivative *= drelu(valuesBeforeActivations[i][layer][0]);
              break;
            default:
              break;

          }

          if (layer != coordinates.key + 1) {
            
            sampleDerivative *= weights[connections[layer][0]][connections[layer-1][0]];

          }

        }

       // sampleDerivative *= weights[coordinates.key][coordinates.value];
        derivative += sampleDerivative;

      }

    return derivative;

  }

  static double gradientDescent(double weight, double learningRate, double dw_ij) {

    //System.out.println(weight + " - " + learningRate + " * " + dw_ij);

    return weight - learningRate * dw_ij;

  }

}