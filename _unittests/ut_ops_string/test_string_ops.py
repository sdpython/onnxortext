"""
@brief      test log(time=2s)
"""
import io
import json
import unittest
import numpy
from onnx import helper, onnx_pb as onnx_proto
import onnxruntime as _ort
from onnxortext import (
    get_library_path as _get_library_path, __domain__ as DOMAIN)
from pyquickhelper.pycode import ExtTestCase


def _create_test_model_test(domain=DOMAIN):
    nodes = []
    nodes.append(helper.make_node(
        'CustomOpOne', ['input_1', 'input_2'], ['output_1'],
        domain=domain))
    nodes.append(helper.make_node(
        'CustomOpTwo', ['output_1'], ['output'],
        domain=domain))

    input0 = helper.make_tensor_value_info(
        'input_1', onnx_proto.TensorProto.FLOAT, [3, 5])
    input1 = helper.make_tensor_value_info(
        'input_2', onnx_proto.TensorProto.FLOAT, [3, 5])
    output0 = helper.make_tensor_value_info(
        'output', onnx_proto.TensorProto.INT32, [3, 5])

    graph = helper.make_graph(nodes, 'test0', [input0, input1], [output0])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid(domain, 1)])
    return model


def _create_test_model_string_equal(domain=DOMAIN):
    assert domain
    nodes = []
    nodes.append(helper.make_node('Identity', ['x'], ['id1']))
    nodes.append(helper.make_node('Identity', ['y'], ['id2']))
    nodes.append(
        helper.make_node('StringEqual', ['id1', 'id2'], ['z'], domain=domain))

    input0 = helper.make_tensor_value_info(
        'x', onnx_proto.TensorProto.STRING, [])
    input1 = helper.make_tensor_value_info(
        'y', onnx_proto.TensorProto.STRING, [])
    output0 = helper.make_tensor_value_info(
        'z', onnx_proto.TensorProto.BOOL, [])

    graph = helper.make_graph(nodes, 'test0', [input0, input1], [output0])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid(domain, 1)])
    return model


def _create_test_model_string_split(domain=DOMAIN):
    assert domain
    nodes = []
    nodes.append(helper.make_node('Identity', ['input'], ['id1']))
    nodes.append(helper.make_node('Identity', ['delimiter'], ['id2']))
    nodes.append(helper.make_node('Identity', ['skip_empty'], ['id3']))
    nodes.append(
        helper.make_node(
            'StringSplit', ['id1', 'id2', 'id3'],
            ['indices', 'values', 'shape'], domain=domain))

    input0 = helper.make_tensor_value_info(
        'input', onnx_proto.TensorProto.STRING, [])
    input1 = helper.make_tensor_value_info(
        'delimiter', onnx_proto.TensorProto.STRING, [])
    input2 = helper.make_tensor_value_info(
        'skip_empty', onnx_proto.TensorProto.UINT8, [])
    output0 = helper.make_tensor_value_info(
        'indices', onnx_proto.TensorProto.INT64, [])
    output1 = helper.make_tensor_value_info(
        'values', onnx_proto.TensorProto.STRING, [])
    output2 = helper.make_tensor_value_info(
        'shape', onnx_proto.TensorProto.INT64, [])

    graph = helper.make_graph(nodes, 'test0', [input0, input1, input2],
                              [output0, output1, output2])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid(domain, 1)])
    return model


class TestPythonOpString(ExtTestCase):

    def enumerate_matrix_couples(self):
        for i in range(1, 5):
            shape = (3,) * i
            a = (numpy.random.rand(*shape) * 10).astype(
                numpy.int32).astype(numpy.str)
            yield a, a
            for j in range(i):
                shape2 = list(shape)
                shape2[j] = 1
                b = (numpy.random.rand(*shape2) * 10).astype(
                    numpy.int32).astype(numpy.str)
                yield a, b
                for k in range(j + 1, i):
                    shape3 = list(shape2)
                    shape3[k] = 1
                    b = (numpy.random.rand(*shape3) * 10).astype(
                        numpy.int32).astype(numpy.str)
                    yield a, b

    def test_ut_cpp(self):
        model_onnx = _create_test_model_test()
        self.assertIn(DOMAIN, str(model_onnx))
        with open("t.onnx", "wb") as f:
            f.write(model_onnx.SerializeToString())

    def test_string_equal_cc(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_equal()
        self.assertIn('op_type: "StringEqual"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)

        for x, y in self.enumerate_matrix_couples():
            txout = sess.run(None, {'x': x, 'y': y})
            self.assertEqual(txout[0].tolist(), (x == y).tolist())
            txout = sess.run(None, {'x': y, 'y': x})
            self.assertEqual(txout[0].tolist(), (y == x).tolist())

    def test_string_split_cc(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_split()
        self.assertIn('op_type: "StringSplit"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input = numpy.array(["a,,b", "", "aa,b,c", "dddddd"])
        delimiter = numpy.array([","])

        for skip in [True, False]:
            with self.subTest(skip=skip):
                skip_empty = numpy.array([skip], dtype=numpy.uint8)

                txout = sess.run(
                    None, {'input': input, 'delimiter': delimiter,
                           'skip_empty': skip_empty})

                if skip_empty:
                    exp_indices = numpy.array(
                        [[0, 0], [0, 1], [2, 0], [2, 1], [2, 2], [3, 0]])
                    exp_text = numpy.array(
                        ['a', 'b', 'aa', 'b', 'c', 'dddddd'])
                else:
                    exp_indices = numpy.array(
                        [[0, 0], [0, 1], [0, 2], [2, 0], [2, 1],
                         [2, 2], [3, 0]])
                    exp_text = numpy.array(
                        ['a', '', 'b', 'aa', 'b', 'c', 'dddddd'])
                exp_shape = numpy.array([4, 3])
                self.assertEqual(exp_indices.tolist(), txout[0].tolist())
                self.assertEqual(exp_text.tolist(), txout[1].tolist())
                self.assertEqual(exp_shape.tolist(), txout[2].tolist())

    def test_string_split_cc_sep2(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_split()
        self.assertIn('op_type: "StringSplit"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input = numpy.array(["a*b", "a,*b", "aa,b,,c", 'z', "dddddd,", "**"])
        delimiter = numpy.array([",*"])

        for skip in [True, False]:
            with self.subTest(skip=skip):
                skip_empty = numpy.array([skip], dtype=numpy.uint8)

                txout = sess.run(
                    None, {'input': input, 'delimiter': delimiter,
                           'skip_empty': skip_empty})

                if skip_empty:
                    exp_indices = numpy.array(
                        [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1],
                         [2, 2], [3, 0], [4, 0]])
                    exp_text = numpy.array(
                        ['a', 'b', 'a', 'b', 'aa', 'b', 'c', 'z', 'dddddd'])
                    exp_shape = numpy.array([6, 3])
                else:
                    exp_indices = numpy.array(
                        [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 0],
                         [2, 1], [2, 2], [2, 3], [3, 0], [4, 0], [4, 1],
                         [5, 0], [5, 1], [5, 2]])
                    exp_text = numpy.array(
                        ['a', 'b', 'a', '', 'b', 'aa', 'b', '', 'c',
                         'z', 'dddddd', '', '', '', ''])
                    exp_shape = numpy.array([6, 4])
                self.assertEqual(exp_text.tolist(), txout[1].tolist())
                self.assertEqual(exp_indices.tolist(), txout[0].tolist())
                self.assertEqual(exp_shape.tolist(), txout[2].tolist())

    def test_string_split_cc_sep0(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_split()
        self.assertIn('op_type: "StringSplit"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input = numpy.array(["a*b", "a,*b"])
        delimiter = numpy.array([""])

        for skip in [True, False]:
            with self.subTest(skip=skip):
                skip_empty = numpy.array([skip], dtype=numpy.uint8)

                txout = sess.run(
                    None, {'input': input, 'delimiter': delimiter,
                           'skip_empty': skip_empty})

                exp_indices = numpy.array(
                    [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [1, 3]])
                exp_text = numpy.array(['a', '*', 'b', 'a', ',', '*', 'b'])
                exp_shape = numpy.array([2, 4])
                self.assertEqual(exp_text.tolist(), txout[1].tolist())
                self.assertEqual(exp_indices.tolist(), txout[0].tolist())
                self.assertEqual(exp_shape.tolist(), txout[2].tolist())


if __name__ == "__main__":
    unittest.main()
