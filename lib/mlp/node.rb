module MLP
  class Node
    attr_accessor :threshold, :weights

    # Takes a threshold_function, which is just an object that
    # responds to #call and takes and returns a float.
    # an array of weights with one for each input,
    # and an optional threshold.
    def initialize(threshold_function, weights)
      @threshold_function = threshold_function
      @weights = weights
    end

    def get_output(input)
      @input = input
      @net = 0.0
      @weights.each_with_index do |weight, i|
        @net += weight * input[i]
      end
      return @threshold_function.call @net
    end
  end
end
