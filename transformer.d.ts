/** TypeScript shape of transformer.json.
 *
 * JSON has no native integer or floating-point distinction. Numeric scalar
 * values are represented as `number`; validate integer fields and dimensions
 * when loading a checkpoint.
 *
 * Matrix dimensions:
 * - token_embedding: [vocabulary_size][model_width]
 * - position_embedding: [maximum_sequence_length][model_width]
 * - attention weights: [model_width][model_width]
 * - first_feedforward_weight: [model_width][feedforward_width]
 * - second_feedforward_weight: [feedforward_width][model_width]
 * - biases: [feedforward_width], [model_width], [vocabulary_size]
 */

export type Vector = number[];
export type Matrix = Vector[];

export interface TransformerCheckpoint {
  vocabulary_size: number;
  model_width: number;
  feedforward_width: number;
  maximum_sequence_length: number;
  random_state: number;
  token_embedding: Matrix;
  position_embedding: Matrix;
  query_weight: Matrix;
  key_weight: Matrix;
  value_weight: Matrix;
  output_weight: Matrix;
  first_feedforward_weight: Matrix;
  second_feedforward_weight: Matrix;
  first_feedforward_bias: Vector;
  second_feedforward_bias: Vector;
  output_bias: Vector;
}
