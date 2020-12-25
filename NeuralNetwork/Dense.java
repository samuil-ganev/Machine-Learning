class Dense {

  private int numberOfNodes;
  private String activation;

  Dense(int numberOfNodes, String activation) {

    this.numberOfNodes = numberOfNodes;
    this.activation = activation;

  }

  int getNumberOfNodes() {

    return numberOfNodes;

  }

  String getActivation() {

    return this.activation;

  }

}