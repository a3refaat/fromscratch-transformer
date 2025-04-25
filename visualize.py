import plotly.graph_objects as go

class NeuralNetworkVisualizer():
    def __init__(self, input_neural_network):
        self.neural_network = input_neural_network
    
    def visualize_static(self):
        num_layers = len(self.neural_network.layers)
        max_neurons = max(layer.num_neurons for layer in self.neural_network.layers)

        # Dynamically scale layer and neuron spacing
        max_width = 800  # Max plot width in pixels
        max_height = 600  # Max plot height in pixels
        layer_spacing = max_width / (num_layers + 1)
        neuron_spacing = max_height / max(1, max_neurons)  

        edge_traces = []
        node_x, node_y, node_labels = [], [], []
        neuron_positions = {}  

        # Assign neuron positions
        for layer_idx, layer in enumerate(self.neural_network.layers):
            num_neurons = layer.num_neurons
            y_start = -(num_neurons - 1) * neuron_spacing / 2
            x_pos = layer_idx * layer_spacing  # Scale based on number of layers
            

            for neuron_idx in range(num_neurons):
                y_pos = y_start + neuron_idx * neuron_spacing # Scale vertically
                neuron_id = f"L{layer_idx}_N{neuron_idx}"
                neuron_positions[neuron_id] = (x_pos, y_pos)

                node_x.append(x_pos)
                node_y.append(y_pos)
                node_labels.append(f"{layer.__class__.__name__}<br>{layer.num_neurons} neurons")

        # Create edges
        for layer_idx in range(1, num_layers):
            prev_layer = self.neural_network.layers[layer_idx - 1]
            curr_layer = self.neural_network.layers[layer_idx]

            for prev_neuron_idx in range(prev_layer.num_neurons):
                prev_neuron_id = f"L{layer_idx - 1}_N{prev_neuron_idx}"
                x0, y0 = neuron_positions[prev_neuron_id]

                for curr_neuron_idx in range(curr_layer.num_neurons):
                    curr_neuron_id = f"L{layer_idx}_N{curr_neuron_idx}"
                    x1, y1 = neuron_positions[curr_neuron_id]

                    edge_traces.append(go.Scatter(
                        x=[x0, x1], y=[y0, y1],
                        mode="lines", line=dict(width=0.5, color="rgba(0, 0, 200, 0.3)"),
                        hoverinfo="none"
                    ))

        # Create node traces (neurons)
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers",
            marker=dict(size=8, color="lightblue", line=dict(width=1, color="black")),
            text=node_labels,
            hoverinfo="text"
        )

        # Build figure
        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title="Neural Network Visualization",
            showlegend=False,
            hovermode="closest",
            paper_bgcolor="darkgray",  # Entire figure background
            plot_bgcolor="black",  # Inner plot background
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

        fig.show()
            
            
        return