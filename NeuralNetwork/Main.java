class Main {

  public static void main(String[] args) {

    //The library has functionality to work only with regression models at the moment. Expect an update soon!

    double[][] d = {
      {1}, 
      {4}, 
      {16},
      {25},
      {49}
    };

    double[][] data = Functions.scaler(d);

    /* for (double[] a : data) {

      for (double b : a)
        System.out.print(b + " ");

      System.out.println();

    } */

    double[] testY = {1, 2, 4, 5, 7};
    double[] y = Functions.scaler(testY);
    
    Dense[] denses = {
      
      new Dense(data[0].length, "null"),
      new Dense(10, "softmax"),
      new Dense(5, "relu"),
      new Dense(3, "softmax"),
      new Dense(1, "sigmoid")
      
    };
    
    Sequential nn = new Sequential(denses);

    //System.out.println("Graph:");
    
    String[] metrics = {"accuracy"};
    nn.compile(0.1, "mse", metrics);
    
    /* double[][] toPredictpre = {{10, 9}};
    
    double[] predictionpre = nn.predict(toPredictpre);
    
    for (double pr : predictionpre) {
      
      System.out.println("Prediction: " + (int)(pr * Functions.yMax));
      
    } */

    nn.fit(data, y, 30);
    
    double[][] toPredict = {{9}};
    
    double[] prediction = nn.predict(toPredict);
    
    for (double pr : prediction) {
      
      System.out.println("Prediction: " + (int)(pr * Functions.yMax));
      
    }

  }

}