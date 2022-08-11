class Perceptron {
    constructor(n, c) {
        this.weights = new Array(n);
        for (let i = 0; i < this.weights.length; i++) {
        this.weights[i] = random(-1, 1);
        }
        this.c = c; // learning rate/constant
    }
    
    train(inputs, desired) {
        let guess = this.feedforward(inputs);
        let error = desired - guess;
        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] += this.c * error * inputs[i];
        }
    }
    // Guess -1 or 1 based on input values
    feedforward(inputs) {
        let sum = 0;
        for (let i = 0; i < this.weights.length; i++) {
            sum += inputs[i] * this.weights[i];
        }
        return this.activate(sum);
    }   
    activate(sum) {
        if (sum > 0) return 1;
            else return -1;
        }
        // Return weights
        getWeights() {
            return this.weights;
        }
    }