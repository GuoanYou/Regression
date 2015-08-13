package Regression;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.meta.AdditiveRegression;
import weka.classifiers.trees.M5P;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class VariousRegressionModelRun {

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		// DataSource train_data = new
		// DataSource("D:\\1�ع����\\numOfBugsFound_py\\mingmingData\\wekadata\\synapse_3train_zs_java.arff");//��ѵ������
		// DataSource test_data = new
		// DataSource("D:\\1�ع����\\numOfBugsFound_py\\mingmingData\\wekadata\\synapse-1.2_zs_java.arff");//����������
		// String
		// path="D:\\1�ع����\\numOfBugsFound_py\\mingmingData\\wekadata\\wekaData.txt";

		DataSource train_data = new DataSource(
				"D:\\1�ع����\\numOfBugsFound_java\\pde_BugFound.arff");// ��ѵ������
		DataSource test_data = new DataSource(
				"D:\\1�ع����\\numOfBugsFound_java\\mylyn_BugFound.arff");// ����������
		String path = "D:\\1�ع����\\numOfBugsFound_Java\\weka_y_hy.txt";

		// DataSource train_data = new
		// DataSource("D:\\1�ع����\\numOfBugsFound_py\\mingmingData\\notrans\\ant-1.7_py.arff");//��ѵ������
		// DataSource test_data = new
		// DataSource("D:\\1�ع����\\numOfBugsFound_py\\mingmingData\\notrans\\camel-1.6_py.arff");//����������
		// String
		// path="D:\\1�ع����\\numOfBugsFound_py\\mingmingData\\wekadata\\wekaData.txt";

		FileWriter fw = new FileWriter(path);
		BufferedWriter bfw = new BufferedWriter(fw);

		Instances insTrain = train_data.getDataSet();
		Instances insTest = test_data.getDataSet();

		insTrain.setClassIndex(insTrain.numAttributes() - 1);// ����ѵ�����У�target��index
		insTest.setClassIndex(insTest.numAttributes() - 1);// ���ò��Լ��У�target��index

		// AdditiveRegression Gradient Boosting
		// Linear Regression
		// RandomForest
		// M5P DecisionTree
		// SMOreg

		Classifier m_classifier = null;
		for (int i = 0; i < 5; i++) {
			if (i == 0)
				m_classifier = new LinearRegression();// ���������������
			if (i == 1)
				m_classifier = new RandomForest();
			if (i == 2)
				m_classifier = new AdditiveRegression();
			if (i == 3)
				m_classifier = new M5P();
			if (i == 4)
				m_classifier = new SMOreg();

			m_classifier.buildClassifier(insTrain);// ѵ��������
			Evaluation eval = new Evaluation(insTrain);
			eval.evaluateModel(m_classifier, insTest);// ����Ч��

			for (int j = 0; j < insTest.size(); j++) {
				// System.out.println(insTest.instance(j).value(0)+" : "+m_classifier.classifyInstance(insTest.instance(j)));
				bfw.write(insTest.instance(j)
						.value(insTest.numAttributes() - 1)
						+ " : "
						+ m_classifier.classifyInstance(insTest.instance(j))
						+ "\n");
			}
			bfw.write("0:0:0" + "\n");
			// eval.crossValidateModel(m_classifier, insTrain, 10, new
			// Random(4));

			// RMSE�Ȳ�Ҫ�����
			System.out.println(eval.rootMeanSquaredError());// ����

		}
		bfw.flush();
		bfw.close();
		// eval.evaluateModel(m_classifier, insTest);//����Ч��
		// System.out.println(eval.toSummaryString());//����
	}

}