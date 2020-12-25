import java.util.Arrays;
import java.util.ArrayList;

class Sequential {

  double learningRate = 0.05;
  String loss;
  String[] metrics;

  private Dense[] denses;
  private int[][] connections;
  private double[][] weights, biases;

  Sequential(Dense[] denses) {

    this.denses = denses;
    int totalNumberOfNodes = getTotalNumberOfNodes();
    
    weights = new double[totalNumberOfNodes][totalNumberOfNodes];
    biases = new double[denses.length][totalNumberOfNodes];
    connections = new int[denses.length][totalNumberOfNodes];

    initializeConnections();
    initializeWeightsAndBiases();

    //print();

  }

  int getTotalNumberOfNodes() {

    int totalNumberOfNodes = 0;

    for (Dense dense : denses)
      totalNumberOfNodes += dense.getNumberOfNodes();

    return totalNumberOfNodes;

  }

  void initializeConnections() {

    for (int[] layer : connections)
      Arrays.fill(layer, -1);

    int layer = 0, numberOfNode = 0;

    for (Dense dense : denses) {

      int nodesInDense = dense.getNumberOfNodes();

      for (int i=0;i<nodesInDense;++i) {

        connections[layer][i] = numberOfNode;
        numberOfNode++;

      }

      layer++;

    }

  }

  void initializeWeightsAndBiases() {

    for (double[] layer : weights)
      Arrays.fill(layer, -1);

    for (double[] layer : biases)
      Arrays.fill(layer, -1);

    int layers = denses.length;

    for (int layer=1;layer<layers;++layer) {

      int currentNodes = connections[layer].length;
      int previousNodes = connections[layer-1].length;

      for (int i=0;i<currentNodes;++i) {

        for (int j=0;j<previousNodes;++j) {

          try {

            weights[connections[layer][i]][connections[layer-1][j]] = Math.random() * 10;
            weights[connections[layer-1][j]][connections[layer][i]] = weights[connections[layer][i]][connections[layer-1][j]];

            biases[layer][i] = 0;

          } catch (Exception e) {}

        }

      }

    }

  }

  void print() {

    int totalNumberOfNodes = getTotalNumberOfNodes();

    System.out.println("Weights:");

    for (int i=0;i<totalNumberOfNodes;++i) {

      for (int j=0;j<totalNumberOfNodes;++j) {

        System.out.print(weights[i][j] + " ");

      }

      System.out.println();

    }

    System.out.println("Biases:");

    for (int i=0;i<denses.length;++i) {

      for (int j=0;j<totalNumberOfNodes;++j) {

        System.out.print(biases[i][j] + " ");

      }

      System.out.println();

    }

    System.out.println("Connections:");

    for (int i=0;i<denses.length;++i) {

      for (int j=0;j<totalNumberOfNodes;++j) {

        System.out.print(connections[i][j] + " ");

      }

      System.out.println();

    }

  }

  void compile(double learningRate, String loss, String[] metrics) {

    this.learningRate = learningRate;
    this.loss = loss;
    this.metrics = metrics;

  }

  void backpropagation(double[][][] values, double[][][] valuesBeforeActivations, double[] yPredicted, double[] y) {

    //mse 
    
    int totalNumberOfNodes = getTotalNumberOfNodes();

    ArrayList<Pair> covered = new ArrayList<Pair>();

    for (int i=0;i<totalNumberOfNodes;++i) {

      for (int j=0;j<totalNumberOfNodes;++j) {

        if (weights[i][j] == -1)
          continue;

        if (Functions.alreadyCovered(covered, new Pair(i, j)))
          continue;

        covered.add(new Pair(i, j));

        double dw_ij = Functions.calculateDerivativeW(new Pair(i, j), this.weights, values, valuesBeforeActivations, this.connections, yPredicted, y, this.denses);

        this.weights[i][j] = Functions.gradientDescent(this.weights[i][j], this.learningRate, dw_ij);
        
        this.weights[j][i] = this.weights[i][j];

      }

    }
    
    for (int i=0;i<denses.length;++i) {
      
      for (int j=0;j<totalNumberOfNodes;++j) {
        
        if (biases[i][j] == -1) 
          continue;
          
        biases[i][j] = Functions.gradientDescent(biases[i][j], this.learningRate, Functions.calculateDerivativeB(new Pair(i, j), this.weights, values, valuesBeforeActivations, this.connections, yPredicted, y, this.denses));
        
      }
      
    }

  }

  void fit(double[][] features, double[] labels, int epochs) {

    //regression

    int totalNumberOfNodes = getTotalNumberOfNodes();

    for (;epochs!=0;epochs--) {

      double[] yPredicted = new double[labels.length];

      double[][][] values = new double[features.length][denses.length][totalNumberOfNodes];
      double[][][] valuesBeforeActivations = new double[features.length][denses.length][totalNumberOfNodes];

      for (int vector=0;vector<features.length;++vector) {

        for (int i=0;i<denses[0].getNumberOfNodes();++i)
          values[vector][0][i] = features[vector][i];

        for (int layer=1;layer<denses.length;++layer) {

          for (int node=0;true;++node) {

            if (connections[layer][node] == -1)
              break;

            for (int previousNodes=0;true;++previousNodes) {

              if (connections[layer-1][previousNodes] == -1)
                break;
                
              values[vector][layer][node] += values[vector][layer-1][previousNodes] * weights[connections[layer][node]][connections[layer-1][previousNodes]];

            }

            values[vector][layer][node] += biases[layer][node];
            valuesBeforeActivations[vector][layer][node] = values[vector][layer][node];

             switch (denses[layer].getActivation()) {

              case "relu":
                values[vector][layer][node] = Functions.relu(values[vector][layer][node]);
                break;
              case "sigmoid":
                values[vector][layer][node] = Functions.sigmoid(values[vector][layer][node]);
                break;
              case "softplus":
                values[vector][layer][node] = Functions.softplus(values[vector][layer][node]);
                break;
              default:
                break;

            }

          }

        }

        yPredicted[vector] = values[vector][denses.length-1][0];

        /* System.out.println("Values: ");

        for (int i=0;i<features.length;++i) {

          for (int j=0;true;++j) {

            if (connections[i][j] == -1)
              break;

            System.out.print(values[vector][i][j] + " ");

          }

          System.out.println();

        } */

      }

      switch (this.loss) {

        case "mse":
          System.out.println("mse: " + Functions.MSE(yPredicted, labels));
          break;
        case "cross-entropy":
          System.out.println("cross-entropy: soon");
          break;
        default:
          break;

      }

      backpropagation(values, valuesBeforeActivations, yPredicted, labels);

      /* System.out.println("Weights: ");

      for (int i=0;i<totalNumberOfNodes;++i) {

        for (int j=0;j<totalNumberOfNodes;++j) {

          System.out.print(weights[i][j] + " ");

        }

        System.out.println();

      } */

    }

    //print();

  }
  
  double[] predict(double[][] features) {
    
    int totalNumberOfNodes = getTotalNumberOfNodes();
    
    double[] yPredicted = new double[features.length];
    
    double[][][] values = new double[features.length][denses.length][totalNumberOfNodes];
    
    for (int vector=0;vector<features.length;++vector) {
    
      for (int i=0;i<denses[0].getNumberOfNodes();++i) {
      
        values[vector][0][i] = features[vector][i];
      
      }
      
      for (int layer=1;layer<denses.length;++layer) {
        
        for (int node=0;true;++node) {
          
          if (connections[layer][node] == -1){
            break;}
            
          for (int previousNodes=0;true;++previousNodes) {
            
            if (connections[layer-1][previousNodes] == -1) { break; }
            
            values[vector][layer][node] += values[vector][layer-1][previousNodes] * weights[connections[layer][node]][connections[layer-1][previousNodes]];
            
          }
          
          values[vector][layer][node] += biases[layer][node];
          
          switch (denses[layer].getActivation()) {

            case "relu":
              values[vector][layer][node] = Functions.relu(values[vector][layer][node]);
                break;
            case "sigmoid":
              values[vector][layer][node] = Functions.sigmoid(values[vector][layer][node]);
                break;
            case "softplus":
              values[vector][layer][node] = Functions.softplus(values[vector][layer][node]);
                break;
            default:
              break;

          }
          
        }
        
      }
      
      yPredicted[vector] = values[vector][denses.length-1][0];
      
    }
    
    return yPredicted;
    
  }

}