import json


class ToyTransformer:
    """A small causal language model with every operation exposed."""

    def __init__(self, vocabulary_size, model_width=8, feedforward_width=16,
                 maximum_sequence_length=32, seed=1):
        self.vocabulary_size = vocabulary_size
        self.model_width = model_width
        self.feedforward_width = feedforward_width
        self.maximum_sequence_length = maximum_sequence_length

        # A tiny deterministic pseudo-random generator keeps this file
        # dependency-free and makes experiments repeatable.
        self.random_state = seed
        self.token_embedding = self.matrix(vocabulary_size, model_width)
        self.position_embedding = self.matrix(maximum_sequence_length,
                                               model_width)
        self.query_weight = self.matrix(model_width, model_width)
        self.key_weight = self.matrix(model_width, model_width)
        self.value_weight = self.matrix(model_width, model_width)
        self.output_weight = self.matrix(model_width, model_width)
        self.first_feedforward_weight = self.matrix(model_width,
                                                     feedforward_width)
        self.second_feedforward_weight = self.matrix(feedforward_width,
                                                      model_width)
        self.first_feedforward_bias = [0.0] * feedforward_width
        self.second_feedforward_bias = [0.0] * model_width
        self.output_bias = [0.0] * vocabulary_size

    @classmethod
    def from_json(cls, path):
        """Create a model by loading the parameter object from a JSON file."""
        with open(path, "r") as file:
            state = json.load(file)

        model = cls.__new__(cls)
        model.vocabulary_size = state["vocabulary_size"]
        model.model_width = state["model_width"]
        model.feedforward_width = state["feedforward_width"]
        model.maximum_sequence_length = state["maximum_sequence_length"]
        model.random_state = state["random_state"]
        model.token_embedding = state["token_embedding"]
        model.position_embedding = state["position_embedding"]
        model.query_weight = state["query_weight"]
        model.key_weight = state["key_weight"]
        model.value_weight = state["value_weight"]
        model.output_weight = state["output_weight"]
        model.first_feedforward_weight = state["first_feedforward_weight"]
        model.second_feedforward_weight = state["second_feedforward_weight"]
        model.first_feedforward_bias = state["first_feedforward_bias"]
        model.second_feedforward_bias = state["second_feedforward_bias"]
        model.output_bias = state["output_bias"]
        model.output_embedding = state.get("output_embedding", model.token_embedding)
        return model

    def next_random(self):
        """Return a repeatable number in approximately [-0.1, 0.1]."""
        self.random_state = (1103515245 * self.random_state + 12345) % 2147483648
        return (self.random_state / 2147483648.0 - 0.5) * 0.2

    def matrix(self, rows, columns):
        """Make a small random matrix using only nested lists."""
        return [[self.next_random() for _ in range(columns)]
                for _ in range(rows)]

    def dot(self, left, right):
        """Multiply a vector by a matrix: [width] x [width][new_width]."""
        return [sum(left[row] * right[row][column]
                    for row in range(len(left)))
                for column in range(len(right[0]))]

    def add(self, left, right):
        return [a + b for a, b in zip(left, right)]

    def relu(self, values):
        return [value if value > 0.0 else 0.0 for value in values]

    def layer_norm(self, values):
        """Normalize one vector.  The small constant avoids division by zero."""
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        scale = (variance + 1e-5) ** 0.5
        return [(value - mean) / scale for value in values]

    def exponential(self, value):
        """Approximate e**value with a short Taylor series.

        Attention only calls this with value <= 0 after the stable shift, so
        this small approximation is adequate for a toy implementation.
        """
        result = 1.0
        term = 1.0
        for order in range(1, 16):
            term *= value / order
            result += term
        return result

    def softmax(self, values):
        """Turn scores into probabilities, with a simple stable shift."""
        highest = max(values)
        exponentials = [self.exponential(value - highest) for value in values]
        total = sum(exponentials)
        return [value / total for value in exponentials]

    def forward(self, token_ids):
        """Return logits for each position, attending only to the past.

        token_ids is a list such as [2, 5, 5].  The result is shaped
        [sequence_length][vocabulary_size].
        """
        if not token_ids:
            return []
        if len(token_ids) > self.maximum_sequence_length:
            raise ValueError("sequence is longer than the configured limit")
        for token_id in token_ids:
            if token_id < 0 or token_id >= self.vocabulary_size:
                raise ValueError("token ID is outside the vocabulary")

        hidden = [self.add(self.token_embedding[token_id],
                           self.position_embedding[position])
                  for position, token_id in enumerate(token_ids)]
        hidden = [self.layer_norm(vector) for vector in hidden]

        queries = [self.dot(vector, self.query_weight) for vector in hidden]
        keys = [self.dot(vector, self.key_weight) for vector in hidden]
        values = [self.dot(vector, self.value_weight) for vector in hidden]

        attended = []
        for position, query in enumerate(queries):
            scores = [sum(query[index] * keys[past][index]
                          for index in range(self.model_width))
                      / (self.model_width ** 0.5)
                      for past in range(position + 1)]
            probabilities = self.softmax(scores)
            context = [sum(probabilities[past] * values[past][index]
                           for past in range(position + 1))
                       for index in range(self.model_width)]
            attended.append(self.add(hidden[position],
                                     self.dot(context, self.output_weight)))

        block_output = []
        for vector in attended:
            normalized = self.layer_norm(vector)
            feedforward = self.dot(normalized,
                                   self.first_feedforward_weight)
            feedforward = self.add(feedforward, self.first_feedforward_bias)
            feedforward = self.relu(feedforward)
            feedforward = self.dot(feedforward,
                                   self.second_feedforward_weight)
            feedforward = self.add(feedforward, self.second_feedforward_bias)
            block_output.append(self.layer_norm(self.add(normalized,
                                                         feedforward)))

        output_embedding = getattr(self, "output_embedding", self.token_embedding)
        return [
            [sum(vector[index] * output_embedding[token][index]
                 for index in range(self.model_width)) + self.output_bias[token]
             for token in range(self.vocabulary_size)]
            for vector in block_output
        ]

    def predict_next(self, token_ids):
        """Return the most likely next token ID according to the last row."""
        logits = self.forward(token_ids)[-1]
        return max(range(self.vocabulary_size), key=lambda token: logits[token])

    def generate(self, token_ids, number_of_tokens):
        """Greedily append tokens; sampling is intentionally left to the reader."""
        result = list(token_ids)
        for _ in range(number_of_tokens):
            context = result[-self.maximum_sequence_length:]
            result.append(self.predict_next(context))
        return result


if __name__ == "__main__":
    model = ToyTransformer.from_json("transformer-poking/transformer.json")
    prompt = [0, 1, 2]
    print("prompt:", prompt)
    print("next token:", model.predict_next(prompt))
    print("generated:", model.generate(prompt, number_of_tokens=5))
