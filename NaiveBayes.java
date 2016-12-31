import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import weka.core.Instances;

public class NaiveBayes {
	static int numOfAttributes;
	public static void main(String[] args) throws IOException {
		if(args.length != 3) {
			System.out.println(args.length);
			System.err.println("Usage: Java DecisionTree"
					+ "<train-set-file> <test-set-file> <n|t>");
			System.exit(1);;
		}
		BufferedReader train = new BufferedReader(
				new FileReader(args[0]));
		BufferedReader test = new BufferedReader(
				new FileReader(args[1]));
		Instances trainSet = new Instances(train);		
		Instances testSet = new Instances(test);
		trainSet.setClassIndex(trainSet.numAttributes() - 1);
		testSet.setClassIndex(testSet.numAttributes() - 1);

		if (args[2].equals("n")) {
			NaiveBayesImpl nb = new NaiveBayesImpl(trainSet,testSet);
			nb.print();
		}
		if (args[2].equals("t")) {
			Tan t = new Tan(trainSet,testSet);
			t.print();
		}

		train.close();
		test.close();

	}

}
//Randomize r = new Randomize();
//int seed = r.getRandomSeed();
//Random rand = new Random(seed);   // create seeded number generator
//Instances randData = new Instances(trainSet);   // create copy of original data
//randData.randomize(rand); 
