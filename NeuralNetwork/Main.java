class Main {

  public static void main(String[] args) {

    //The library has functionality to work only with regression models at the moment. Expect an update soon!

    double[][] d = {
      {1}, 
      {3}, 
      {5},
      {6},
      {4},
      {0}
    };

    double[][] data = Functions.scaler(d);

    /* for (double[] a : data) {

      for (double b : a)
        System.out.print(b + " ");

      System.out.println();

    } */

    double[] testY = {2, 4, 6, 7, 5, 1};
    double[] y = Functions.scaler(testY);
    
    Dense[] denses = {
      
      new Dense(data[0].length, "null"),
      new Dense(10, "softmax"),
      new Dense(5, "relu"),
      new Dense(3, "softmax"),
      new Dense(1, "sigmoid")
      
    };
    
    Sequential nn = new Sequential(denses);

    System.out.println("Graph:");
    
    String[] metrics = {"accuracy"};
    nn.compile(1, "mse", metrics);
    
    double[][] toPredictpre = {{4}};
    
    double[] predictionpre = nn.predict(toPredictpre);
    
    for (double pr : predictionpre) {
      
      System.out.println("Prediction: " + pr * Functions.yMax);
      
    }

    nn.fit(data, y, 10);
    
    double[][] toPredict = {{4}};
    
    double[] prediction = nn.predict(toPredict);
    
    for (double pr : prediction) {
      
      System.out.println("Prediction: " + pr * Functions.yMax);
      
    }

  }

}