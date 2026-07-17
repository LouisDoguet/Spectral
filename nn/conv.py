"""Manually construct ONNX graph for NodalAlphaModel with proper weights."""

import numpy as np
import onnx
from onnx import helper, TensorProto


def create_nodal_alpha_onnx(
    P: int,
    width: int = 16,
    n_elem: int = 32,
    output_path: str = "nodal_alpha_model.onnx",
):
    """
    Manually create an ONNX graph representing NodalAlphaModel.
    Includes dummy weights for visualization.
    """
    
    Nn = P + 1
    in_channels = 1 + Nn
    
    # ===== INPUT/OUTPUT DEFINITIONS =====
    input_tensor = helper.make_tensor_value_info(
        name="input",
        elem_type=TensorProto.FLOAT,
        shape=[1 + Nn, n_elem * Nn],
    )
    
    output_tensor = helper.make_tensor_value_info(
        name="alpha",
        elem_type=TensorProto.FLOAT,
        shape=[n_elem, P],
    )
    
    # ===== INITIALIZERS (weights and biases) =====
    initializers = []
    
    # Lift conv: (1+Nn) -> width, kernel_size=3
    lift_w = np.random.randn(width, in_channels, 3).astype(np.float32)
    lift_b = np.zeros(width, dtype=np.float32)
    initializers.append(helper.make_tensor("lift_w", TensorProto.FLOAT, lift_w.shape, lift_w.tobytes(), raw=True))
    initializers.append(helper.make_tensor("lift_b", TensorProto.FLOAT, lift_b.shape, lift_b.tobytes(), raw=True))
    
    # ResBlock conv1: width -> width
    res1_w = np.random.randn(width, width, 3).astype(np.float32)
    res1_b = np.zeros(width, dtype=np.float32)
    initializers.append(helper.make_tensor("res1_w", TensorProto.FLOAT, res1_w.shape, res1_w.tobytes(), raw=True))
    initializers.append(helper.make_tensor("res1_b", TensorProto.FLOAT, res1_b.shape, res1_b.tobytes(), raw=True))
    
    # ResBlock conv2: width -> width
    res2_w = np.random.randn(width, width, 3).astype(np.float32)
    res2_b = np.zeros(width, dtype=np.float32)
    initializers.append(helper.make_tensor("res2_w", TensorProto.FLOAT, res2_w.shape, res2_w.tobytes(), raw=True))
    initializers.append(helper.make_tensor("res2_b", TensorProto.FLOAT, res2_b.shape, res2_b.tobytes(), raw=True))
    
    # Readout conv: width -> 1
    readout_w = np.random.randn(1, width, 3).astype(np.float32)
    readout_b = np.zeros(1, dtype=np.float32)
    initializers.append(helper.make_tensor("readout_w", TensorProto.FLOAT, readout_w.shape, readout_w.tobytes(), raw=True))
    initializers.append(helper.make_tensor("readout_b", TensorProto.FLOAT, readout_b.shape, readout_b.tobytes(), raw=True))
    
    # ===== NODES (operations) =====
    nodes = []
    
    # 1. Lift Conv1d: input -> lift_out
    nodes.append(helper.make_node(
        op_type="Conv",
        inputs=["input", "lift_w", "lift_b"],
        outputs=["lift_conv_out"],
        kernel_shape=[3],
        pads=[1, 1],  # same padding
        name="lift_conv",
    ))
    
    # 2. Lift ReLU
    nodes.append(helper.make_node(
        op_type="Relu",
        inputs=["lift_conv_out"],
        outputs=["lift_relu_out"],
        name="lift_relu",
    ))
    
    # 3. ResBlock Conv1
    nodes.append(helper.make_node(
        op_type="Conv",
        inputs=["lift_relu_out", "res1_w", "res1_b"],
        outputs=["res1_conv_out"],
        kernel_shape=[3],
        pads=[1, 1],
        name="resblock_conv1",
    ))
    
    # 4. ResBlock Conv1 ReLU
    nodes.append(helper.make_node(
        op_type="Relu",
        inputs=["res1_conv_out"],
        outputs=["res1_relu_out"],
        name="resblock_relu1",
    ))
    
    # 5. ResBlock Conv2
    nodes.append(helper.make_node(
        op_type="Conv",
        inputs=["res1_relu_out", "res2_w", "res2_b"],
        outputs=["res2_conv_out"],
        kernel_shape=[3],
        pads=[1, 1],
        name="resblock_conv2",
    ))
    
    # 6. Skip connection: add res2_conv_out + lift_relu_out
    nodes.append(helper.make_node(
        op_type="Add",
        inputs=["res2_conv_out", "lift_relu_out"],
        outputs=["skip_add_out"],
        name="skip_add",
    ))
    
    # 7. Skip ReLU
    nodes.append(helper.make_node(
        op_type="Relu",
        inputs=["skip_add_out"],
        outputs=["skip_relu_out"],
        name="skip_relu",
    ))
    
    # 8. Readout Conv (width -> 1)
    nodes.append(helper.make_node(
        op_type="Conv",
        inputs=["skip_relu_out", "readout_w", "readout_b"],
        outputs=["readout_conv_out"],
        kernel_shape=[3],
        pads=[1, 1],
        name="readout_conv",
    ))
    
    # 9. Sigmoid (for alpha in (0, 1))
    nodes.append(helper.make_node(
        op_type="Sigmoid",
        inputs=["readout_conv_out"],
        outputs=["alpha"],
        name="sigmoid",
    ))
    
    # ===== CREATE GRAPH =====
    graph = helper.make_graph(
        nodes=nodes,
        name="NodalAlphaModel",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=initializers,
    )
    
    # ===== CREATE MODEL =====
    model = helper.make_model(
        graph,
        producer_name="NodalAlphaModel",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    
    # ===== SAVE =====
    try:
        onnx.checker.check_model(model)
        print(f"✅ ONNX model is valid")
    except Exception as e:
        print(f"⚠️  Model check failed: {e}")
        print("   (This might be OK for visualization, proceeding anyway...)")
    
    onnx.save(model, output_path)
    print(f"✅ ONNX model saved to {output_path}")
    
    return output_path


if __name__ == "__main__":
    create_nodal_alpha_onnx(P=4, width=16, n_elem=32, output_path="nodal_alpha_model.onnx")
    