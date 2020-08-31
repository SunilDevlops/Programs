package com.array;
public class FindLargestAndSecondLargestInArray {
    public static void main(String... str) {
        int[] arr = {50,50,50,50};
        int n = arr.length;
        int max = arr[0];
        int big;
        int secondLargest = arr[0];
        for(int i = 0; i < n; i = i+2){
            if(arr[i] > arr[i+1]) {
                big = arr[i];
            } else {
                big = arr[i+1];
            }
            if(big > max) {
                secondLargest = max;
                max = big;
            }
        }
        if(largest = secondLargest) {
            secondLargest = null;
        }
        System.out.println("The largest number in unsorted array is : "+ max);
        if(secondLargest!=null)
            System.out.println("The second largest number in unsorted array is : " + secondLargest);
    }
}