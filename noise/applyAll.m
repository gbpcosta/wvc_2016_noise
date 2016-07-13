function applyAll(pathImages,typeImage)

  addNoise(pathImages,typeImage,'gaussian',10);
  addNoise(pathImages,typeImage,'gaussian',20);
  addNoise(pathImages,typeImage,'gaussian',30);
  addNoise(pathImages,typeImage,'gaussian',40);
  addNoise(pathImages,typeImage,'gaussian',50);

  addNoise(pathImages,typeImage,'poisson',2);
  addNoise(pathImages,typeImage,'poisson',4);
  addNoise(pathImages,typeImage,'poisson',6);
  addNoise(pathImages,typeImage,'poisson',8);
  addNoise(pathImages,typeImage,'poisson',10);

  addNoise(pathImages,typeImage,'sp',0.1);
  addNoise(pathImages,typeImage,'sp',0.2);
  addNoise(pathImages,typeImage,'sp',0.3);
  addNoise(pathImages,typeImage,'sp',0.4);
  addNoise(pathImages,typeImage,'sp',0.5);

end
