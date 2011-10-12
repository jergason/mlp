module MLP
  class MLP
    # Takes an array of the form [input_nodes, number_of_hidden_nodes_for_hidden_layer . . ., number_of_output_nodes].
    def initialize(arr)
      #store weights as a 2d array
      raise ArgumentError.new('Input must be an array of at least length 3') unless arr.instance_of? Array
      @input_nodes = arr[0]
      @hidden_layers = arr[1...-1]
      @output_nodes = arr[-1]

      @threshold_function = Proc.new { |net| 1 / (1 + Math.exp(-net)) }
      #build the network
      build_network
    end

    def build_network
      input_layer = []
      @input_nodes.times do
        input_layer << MLP::Node.new(@threshold_function, [0.5])
      end
      @input_nodes = input_layer



      hidden_layers = []
      num_nodes_in_first_hidden_layer = @hidden_layers.unshift
      #is this the best way to do it? Every node in one layer feeds in to the next layer?
      @hidden_layers.each do |number_of_hidden_nodes|
        if layer == 0

        end
      end
    end
  end
end
